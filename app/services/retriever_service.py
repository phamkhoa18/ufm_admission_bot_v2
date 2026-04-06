"""
Retriever Service v2 — Hybrid Search (Vector + BM25 + RRF) + Parent Retrieval.

Cải tiến so với v1:
  ✅ ThreadedConnectionPool thay single connection
  ✅ Dynamic WHERE Builder (gộp SQL duplicate)
  ✅ DictCursor thay row[0], row[1]...
  ✅ Metadata Filter: program_level + program_name
  ✅ Multi-Query Score Boost (4 embeddings)
  ✅ Parallel Vector + BM25 (ThreadPoolExecutor)
  ✅ BM25 dùng stored tsvector column (GIN index)
  ✅ Observability metrics dict
"""

import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.config.retriever import RetrieverConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# DATABASE CONNECTION POOL (ThreadedConnectionPool)
# ══════════════════════════════════════════════════════════
_pool = None


def _get_pool(cfg: RetrieverConfig):
    """
    Lấy connection pool (thread-safe). Tạo mới nếu chưa có.
    """
    global _pool

    if _pool is not None:
        return _pool

    try:
        import psycopg2
        from psycopg2.pool import ThreadedConnectionPool
    except ImportError:
        raise ImportError(
            "psycopg2 chưa được cài. Chạy: pip install psycopg2-binary"
        )

    db = cfg.db
    _pool = ThreadedConnectionPool(
        minconn=db.pool_min,
        maxconn=db.pool_max,
        host=db.host,
        port=db.port,
        dbname=db.dbname,
        user=db.user,
        password=db.password,
        connect_timeout=db.connect_timeout,
        options=f"-c statement_timeout={db.query_timeout * 1000}",
    )
    logger.info("Retriever - Connection Pool: %s:%s/%s (min=%d, max=%d)",
                db.host, db.port, db.dbname, db.pool_min, db.pool_max)
    return _pool


def _get_connection(cfg: RetrieverConfig):
    """Lấy 1 connection từ pool."""
    pool = _get_pool(cfg)
    conn = pool.getconn()
    conn.autocommit = True
    return conn


def _put_connection(conn):
    """Trả connection về pool."""
    if _pool is not None and conn is not None:
        try:
            _pool.putconn(conn)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════
# DYNAMIC WHERE BUILDER (Gộp tất cả filter logic)
# ══════════════════════════════════════════════════════════
def _build_filters(program_level: str = None, program_name: str = None):
    """
    Xây mệnh đề WHERE động cho SQL queries.
    Tránh duplicate code giữa vector / bm25.

    Returns:
        (where_clause: str, params: list)
    """
    clauses = ["is_active = TRUE"]
    params = []

    if program_level:
        clauses.append("(program_level = %s OR program_level IS NULL)")
        params.append(program_level)

    if program_name:
        clauses.append("(program_name ILIKE %s OR program_name IS NULL)")
        params.append(f"%{program_name}%")

    return " AND ".join(clauses), params


# ══════════════════════════════════════════════════════════
# VECTOR SEARCH (pgvector Cosine Similarity — HNSW Index)
# ══════════════════════════════════════════════════════════
def search_vector(
    query_embedding: list,
    cfg: RetrieverConfig,
    program_level: str = None,
    program_name: str = None,
) -> list:
    """
    Tìm Top K chunks có cosine similarity cao nhất với query vector.
    Hỗ trợ metadata filter: program_level + program_name.
    """
    vs_cfg = cfg.vector_search
    if not vs_cfg.enabled:
        return []

    conn = _get_connection(cfg)
    try:
        from psycopg2.extras import RealDictCursor

        vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        where_clause, where_params = _build_filters(program_level, program_name)

        sql = f"""
            SELECT
                chunk_id,
                chunk_level,
                parent_id,
                section_path,
                program_name,
                1 - (embedding <=> %s::vector) AS cosine_score,
                LEFT(content, 200) AS content_preview
            FROM knowledge_chunks
            WHERE {where_clause}
              AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """

        params = [vec_str] + where_params + [vec_str, vs_cfg.top_k]

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "chunk_id": row["chunk_id"],
                "chunk_level": row["chunk_level"],
                "parent_id": row["parent_id"],
                "section_path": row["section_path"],
                "program_name": row["program_name"],
                "score": float(row["cosine_score"]),
                "content_preview": row["content_preview"],
                "source": "vector",
            })

        # Log cảnh báo nếu top 1 dưới ngưỡng (chỉ warning, KHÔNG chặn data)
        if results and results[0]["score"] < vs_cfg.similarity_threshold:
            logger.warning(
                "Retriever - Top1 cosine=%.4f < threshold %.2f",
                results[0]['score'], vs_cfg.similarity_threshold
            )

        return results
    finally:
        _put_connection(conn)


# ══════════════════════════════════════════════════════════
# BM25 FULL-TEXT SEARCH (Postgres tsvector — stored column)
# ══════════════════════════════════════════════════════════
def search_bm25(
    query_text: str,
    cfg: RetrieverConfig,
    program_level: str = None,
    program_name: str = None,
) -> list:
    """
    Tìm Top K chunks bằng BM25 (stored tsvector + ts_rank_cd).
    Dùng cột content_tsvector (đã lưu sẵn) nếu có, fallback sang runtime.
    """
    bm_cfg = cfg.bm25_search
    if not bm_cfg.enabled:
        return []

    conn = _get_connection(cfg)
    try:
        from psycopg2.extras import RealDictCursor

        ts_cfg = bm_cfg.ts_config
        where_clause, where_params = _build_filters(program_level, program_name)

        # Ưu tiên dùng stored tsvector column (nếu đã tạo migration)
        # Fallback về runtime tsvector nếu chưa migrate
        tsvector_col = getattr(bm_cfg, 'use_stored_tsvector', False)

        if tsvector_col:
            # Dùng stored column — GIN index hoạt động
            tsvec_expr = "content_tsvector"
        else:
            # Fallback — tính runtime (chậm hơn)
            tsvec_expr = f"to_tsvector('{ts_cfg}', content)"

        sql = f"""
            SELECT
                chunk_id,
                chunk_level,
                parent_id,
                section_path,
                program_name,
                ts_rank_cd(
                    {tsvec_expr},
                    plainto_tsquery('{ts_cfg}', %s)
                ) AS bm25_score,
                LEFT(content, 200) AS content_preview
            FROM knowledge_chunks
            WHERE {where_clause}
              AND {tsvec_expr} @@ plainto_tsquery('{ts_cfg}', %s)
            ORDER BY bm25_score DESC
            LIMIT %s;
        """

        params = [query_text] + where_params + [query_text, bm_cfg.top_k]

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "chunk_id": row["chunk_id"],
                "chunk_level": row["chunk_level"],
                "parent_id": row["parent_id"],
                "section_path": row["section_path"],
                "program_name": row["program_name"],
                "score": float(row["bm25_score"]),
                "content_preview": row["content_preview"],
                "source": "bm25",
            })

        return results
    finally:
        _put_connection(conn)


# ══════════════════════════════════════════════════════════
# RRF — Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════
def rrf_merge_weighted(
    ranked_lists: list,
    k: int = 60,
) -> list:
    """
    Gộp N danh sách kết quả bằng Weighted Reciprocal Rank Fusion.

    Args:
        ranked_lists: [(results_list, weight), ...]
            - weight = 1.0 cho nguồn bình thường
            - weight = 1.3 cho standalone_query (priority boost)
        k: Hằng số RRF (mặc định 60, paper gốc Cormack et al. 2009)

    Công thức: RRF_score(d) = Σ weight_i / (k + rank_i(d))
    """
    rrf_scores: dict = {}     # chunk_id → total RRF score
    chunk_data: dict = {}     # chunk_id → metadata dict

    for results, weight in ranked_lists:
        for rank, item in enumerate(results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (weight / (k + rank))
            # Giữ bản có cosine score cao nhất (ưu tiên vector > bm25)
            if cid not in chunk_data:
                chunk_data[cid] = item
            elif item.get("source") == "vector" and chunk_data[cid].get("source") != "vector":
                chunk_data[cid] = item

    # Sắp xếp theo RRF score giảm dần
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for cid in sorted_ids:
        entry = chunk_data[cid].copy()
        entry["rrf_score"] = rrf_scores[cid]
        merged.append(entry)

    return merged


# ══════════════════════════════════════════════════════════
# MULTI-QUERY VECTOR BOOST — Tận dụng 4 embeddings
# ══════════════════════════════════════════════════════════
def search_vector_multi_query(
    query_embeddings: list,
    cfg: RetrieverConfig,
    program_level: str = None,
    program_name: str = None,
) -> list:
    """
    Chạy Vector Search SONG SONG với tất cả embeddings (standalone + multi-query).
    Deduplicate và giữ score cao nhất cho mỗi chunk.

    Tối ưu: ThreadPoolExecutor chạy N DB queries cùng lúc thay vì tuần tự.
    """
    best_scores: dict = {}   # chunk_id → best score
    best_data: dict = {}     # chunk_id → metadata dict

    # ── Chạy song song tất cả vector searches ──
    with ThreadPoolExecutor(max_workers=min(len(query_embeddings), 4)) as pool:
        futures = [
            pool.submit(search_vector, emb, cfg, program_level, program_name)
            for emb in query_embeddings
        ]
        for future in as_completed(futures):
            results = future.result()
            for item in results:
                cid = item["chunk_id"]
                if cid not in best_scores or item["score"] > best_scores[cid]:
                    best_scores[cid] = item["score"]
                    best_data[cid] = item

    # Sắp xếp theo cosine score giảm dần, lấy top_k
    sorted_ids = sorted(best_scores.keys(), key=lambda x: best_scores[x], reverse=True)
    top_k = cfg.vector_search.top_k

    return [best_data[cid] for cid in sorted_ids[:top_k]]


# ══════════════════════════════════════════════════════════
# PARENT EXTRACTION — Lấy top 5 Parent IDs duy nhất
# ══════════════════════════════════════════════════════════
def extract_unique_parent_ids(
    ranked_chunks: list,
    top_parents: int = 5,
) -> list:
    """
    Duyệt danh sách RRF đã sắp xếp → Trích ra top N parent_ids duy nhất.

    Chiến lược ưu tiên:
      - Nếu chunk là child → Lấy parent_id của nó.
      - Nếu chunk là parent → Lấy chính chunk_id.
      - Nếu chunk là standard (không cha-con) → Lấy chính chunk_id.
      - Nếu parent_id đã xuất hiện trước đó → BỎ QUA (chống trùng).
      - Dừng khi đủ top_parents IDs.
    """
    seen_parents = set()
    parent_ids = []

    for chunk in ranked_chunks:
        level = chunk.get("chunk_level", "standard")
        chunk_id = chunk["chunk_id"]
        parent_id = chunk.get("parent_id")

        # Xác định ID đại diện
        if level == "child" and parent_id:
            representative_id = parent_id
        else:
            representative_id = chunk_id

        if representative_id in seen_parents:
            continue

        seen_parents.add(representative_id)
        parent_ids.append(representative_id)

        if len(parent_ids) >= top_parents:
            break

    return parent_ids


# ══════════════════════════════════════════════════════════
# FETCH PARENT CONTENT — Lấy nội dung đầy đủ từ DB
# ══════════════════════════════════════════════════════════
def fetch_parent_contents(
    parent_ids: list,
    cfg: RetrieverConfig,
) -> list:
    """
    Query DB lấy nội dung đầy đủ của Parent Chunks.
    Giữ nguyên thứ tự ưu tiên từ danh sách RRF.
    Dùng DictCursor để truy cập theo tên cột (an toàn hơn index).
    """
    if not parent_ids:
        return []

    conn = _get_connection(cfg)
    try:
        from psycopg2.extras import RealDictCursor

        sql = """
            SELECT
                chunk_id,
                content,
                source,
                section_path,
                section_name,
                program_name,
                program_level,
                ma_nganh,
                academic_year,
                chunk_level,
                char_count,
                extra
            FROM knowledge_chunks
            WHERE chunk_id = ANY(%s)
              AND is_active = TRUE
        """

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (parent_ids,))
            rows = cur.fetchall()

        # Map kết quả theo chunk_id để giữ thứ tự ưu tiên
        content_map = {row["chunk_id"]: dict(row) for row in rows}

        # Trả về đúng thứ tự ưu tiên RRF
        return [content_map[pid] for pid in parent_ids if pid in content_map]
    finally:
        _put_connection(conn)


# ══════════════════════════════════════════════════════════
# FORMAT CONTEXT — Đóng gói context cho LLM chính
# ══════════════════════════════════════════════════════════
def format_rag_context(
    parent_docs: list,
    cfg: RetrieverConfig,
) -> str:
    """
    Gộp nội dung parent docs thành 1 chuỗi text có trích dẫn cấu trúc.
    Cắt ngắn nếu tổng vượt max_parent_chars.
    Tạo link đọc tài liệu tạm thời (session ID).
    """
    from app.utils.document_session import create_document_session
    
    pr_cfg = cfg.parent_retrieval
    parts = []
    total_chars = 0

    for i, doc in enumerate(parent_docs, start=1):
        content = doc["content"]

        # Kiểm tra tổng ký tự không vượt giới hạn
        if total_chars + len(content) > pr_cfg.max_parent_chars:
            remaining = pr_cfg.max_parent_chars - total_chars
            if remaining > 100:
                content = content[:remaining] + "..."
            else:
                break

        # Tạo Session Tài liệu Tạm (5 phút)
        # Dùng chunk_id làm session_id (thay vì random UUID)
        chunk_id = doc.get("chunk_id", "")
        doc_session_data = {
            "title": doc.get("extra", {}).get("title") or doc.get("source") or f"Tài liệu {i}",
            "content": doc["content"], # Lưu full nội dung ban đầu không giới hạn char len
            "metadata": doc.get("extra", {})
        }
        session_id = create_document_session(doc_session_data, session_id=str(chunk_id))
        temp_link = f"/view-document/{session_id}"

        # Gắn metadata header chuẩn xác theo format yêu cầu:
        # {{ SOURCE: TÊN FILE
        # PATH (NẰM Ở MỤC:...)
        if pr_cfg.include_metadata:
            extra = doc.get("extra") or {}
            
            # Tên file/văn bản
            doc_title = extra.get("title") or doc.get("source") or "Tài liệu hệ thống"
            
            # Path
            section_path = doc.get("section_path") or "Không rõ"
            formatted_path = section_path.replace(" > ", " > ")
            
            # Bậc đào tạo
            program_level = doc.get("program_level") or "Chung"

            # Render template CHUẨN XÁC
            header = (
                f"{i}.{{{{ SOURCE: {doc_title}\n"
                f"PATH (NẰM Ở MỤC: {formatted_path})\n"
                f"BẬC ĐÀO TẠO: {program_level}\n"
                f"LINK CHI TIẾT DÀNH CHO DOC_VIEWER: [{doc_title}]({temp_link}) }}}}\n\n"
            )
            parts.append(header + content)
        else:
            parts.append(f"{i}.{{{{ LINK ĐỌC FILE CHI TIẾT: {temp_link} }}}}\n\n{content}")

        total_chars += len(content)

    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════
# HÀM CHÍNH: HYBRID RETRIEVAL PIPELINE v2
# ══════════════════════════════════════════════════════════
def hybrid_retrieve(
    query_text: str,
    query_embedding: list,
    cfg: RetrieverConfig = None,
    program_level: str = None,
    program_name: str = None,
    query_embeddings: list = None,
) -> dict:
    """
    🔍 Pipeline Hybrid Search v2 — tối ưu tốc độ + chính xác.

    Args:
        query_text: Câu hỏi standalone (dùng cho BM25).
        query_embedding: Vector 1024D chính (dùng cho Cosine Search).
        cfg: Config object (mặc định tạo mới từ YAML).
        program_level: Filter bậc đào tạo (thac_si/tien_si/dai_hoc).
        program_name: Filter tên ngành (ILIKE match).
        query_embeddings: Danh sách tất cả embeddings (multi-query boost).

    Returns dict:
        {
            "rag_context": str,
            "retrieved_chunks": list,
            "parent_ids": list,
            "parent_docs": list,
            "vector_count": int,
            "bm25_count": int,
            "top1_cosine_score": float,
            "elapsed": float,
            "metrics": dict,
        }
    """
    if cfg is None:
        cfg = RetrieverConfig()

    start_time = time.time()

    # ── Log metadata filter ──
    if program_level:
        logger.info("Retriever - program_level_filter='%s'", program_level)
    if program_name:
        logger.info("Retriever - program_name_filter='%s'", program_name)

    # ════════════════════════════════════════════════════════
    # Bước 1+2: Vector (standalone + variants) + BM25 SONG SONG
    # ════════════════════════════════════════════════════════
    use_multi = (
        query_embeddings
        and len(query_embeddings) > 1
        and getattr(cfg.vector_search, 'use_multi_query', False)
    )

    standalone_boost = cfg.rrf.standalone_boost

    if use_multi:
        # ── Chạy SONG SONG: standalone + từng variant + BM25 ──
        logger.info(
            "Retriever - Multi-Query Vector Search (%d embeddings, top %d each, boost=%.1f)...",
            len(query_embeddings), cfg.vector_search.top_k, standalone_boost,
        )
        with ThreadPoolExecutor(max_workers=min(len(query_embeddings) + 1, 5)) as pool:
            # Mỗi embedding chạy search riêng, mỗi cái lấy top_k
            vec_futures = [
                pool.submit(search_vector, emb, cfg, program_level, program_name)
                for emb in query_embeddings
            ]
            bm25_future = pool.submit(
                search_bm25, query_text, cfg, program_level, program_name,
            )

            vec_results_per_query = [f.result() for f in vec_futures]
            bm25_results = bm25_future.result()

        # Log
        for i, vr in enumerate(vec_results_per_query):
            tag = "standalone" if i == 0 else f"variant_{i}"
            logger.info("Retriever - Vector [%s] tim duoc: %d chunks", tag, len(vr))
        logger.info("Retriever - BM25 tim duoc: %d chunks", len(bm25_results))

        total_vec = sum(len(vr) for vr in vec_results_per_query)

        # ── Bước 3: Weighted RRF Merge ──
        # standalone → boost, variants + BM25 → weight 1.0
        rrf_lists = []
        for i, vr in enumerate(vec_results_per_query):
            weight = standalone_boost if i == 0 else 1.0
            rrf_lists.append((vr, weight))
        rrf_lists.append((bm25_results, 1.0))

        logger.info("Retriever - Weighted RRF Merge (k=%d, standalone_boost=%.1f)...",
                    cfg.rrf.k, standalone_boost)
        merged = rrf_merge_weighted(rrf_lists, k=cfg.rrf.k)

    else:
        # ── Single vector + BM25 (không có biến thể) ──
        with ThreadPoolExecutor(max_workers=2) as pool:
            logger.info("Retriever - Vector Search (top %d)...", cfg.vector_search.top_k)
            vec_future = pool.submit(
                search_vector, query_embedding, cfg, program_level, program_name,
            )
            logger.info("Retriever - BM25 Search (top %d)...", cfg.bm25_search.top_k)
            bm25_future = pool.submit(
                search_bm25, query_text, cfg, program_level, program_name,
            )
            vector_results = vec_future.result()
            bm25_results = bm25_future.result()

        logger.info("Retriever - Vector tim duoc: %d chunks", len(vector_results))
        logger.info("Retriever - BM25 tim duoc: %d chunks", len(bm25_results))
        total_vec = len(vector_results)

        # RRF không boost (chỉ 2 lists)
        merged = rrf_merge_weighted(
            [(vector_results, 1.0), (bm25_results, 1.0)],
            k=cfg.rrf.k,
        )

    if total_vec > 0 and merged:
        top_score = merged[0].get("score") or merged[0].get("rrf_score", 0)
        logger.debug("  Top1 score: %.4f", top_score)

    logger.info("Retriever - Tong unique chunks sau RRF: %d", len(merged))

    # ── Bước 4: Extract Parent IDs ──
    top_n = cfg.parent_retrieval.top_parents
    logger.info("Retriever - Trich xuat Top %d Parent IDs...", top_n)
    parent_ids = extract_unique_parent_ids(merged, top_parents=top_n)
    logger.debug("Retriever - Parent IDs: %s", parent_ids)

    # ── Bước 5: Fetch Parent Contents ──
    parent_docs = fetch_parent_contents(parent_ids, cfg)
    logger.info("Retriever - Tai noi dung: %d parent docs", len(parent_docs))

    # ── Bước 6: Format Context ──
    rag_context = format_rag_context(parent_docs, cfg)

    elapsed = time.time() - start_time
    top1_cosine = merged[0].get("score", 0.0) if merged else 0.0
    logger.info(
        "Retriever - Hybrid Search hoan tat (%.3fs) - %d ky tu, top1=%.4f",
        elapsed, len(rag_context), top1_cosine
    )

    # ════════════════════════════════════════════════════════
    # Observability Metrics
    # ════════════════════════════════════════════════════════
    metrics = {
        "vector_count": total_vec,
        "bm25_count": len(bm25_results),
        "rrf_unique": len(merged),
        "top1_cosine": top1_cosine,
        "parent_docs": len(parent_docs),
        "context_chars": len(rag_context),
        "latency_ms": round(elapsed * 1000, 1),
        "program_level": program_level or "all",
        "program_name": program_name or "all",
        "multi_query": use_multi,
    }

    return {
        "rag_context": rag_context,
        "retrieved_chunks": merged,
        "parent_ids": parent_ids,
        "parent_docs": parent_docs,
        "vector_count": total_vec,
        "bm25_count": len(bm25_results),
        "top1_cosine_score": top1_cosine,
        "elapsed": elapsed,
        "metrics": metrics,
    }
