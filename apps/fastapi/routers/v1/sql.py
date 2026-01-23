"""
Delta Lake SQL Router

Handles SQL queries against Delta Lake tables using DuckDB.
This router is independent of ML training and provides data exploration capabilities.
"""
from fastapi import APIRouter, HTTPException
import asyncio
import time
import os

from models import SQLQueryRequest, TableSchemaRequest
from config import (
    DELTA_PATHS,
    SQL_DEFAULT_LIMIT,
    SQL_MAX_LIMIT,
    SQL_QUERY_TIMEOUT,
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_S3_ENDPOINT,
)

# Disable EC2 metadata lookup to prevent timeouts when running locally
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

router = APIRouter()


# =============================================================================
# DuckDB Connection Management
# =============================================================================
_duckdb_connection = None


def _get_duckdb_connection(force_reconnect: bool = False):
    """Get or create a DuckDB connection with Delta Lake and S3 support.

    Args:
        force_reconnect: If True, close existing connection and create new one
    """
    global _duckdb_connection

    if force_reconnect and _duckdb_connection is not None:
        try:
            _duckdb_connection.close()
        except Exception:
            pass
        _duckdb_connection = None

    if _duckdb_connection is None:
        import duckdb
        _duckdb_connection = duckdb.connect()
        _duckdb_connection.execute("INSTALL delta; LOAD delta;")
        _duckdb_connection.execute("INSTALL httpfs; LOAD httpfs;")
        # Configure S3/MinIO credentials using CREATE SECRET (DuckDB recommended approach)
        _duckdb_connection.execute(f"""
            CREATE SECRET IF NOT EXISTS minio_secret (
                TYPE S3,
                KEY_ID '{AWS_ACCESS_KEY_ID}',
                SECRET '{AWS_SECRET_ACCESS_KEY}',
                REGION '{AWS_REGION}',
                ENDPOINT '{AWS_S3_ENDPOINT}',
                URL_STYLE 'path',
                USE_SSL false
            );
        """)
    return _duckdb_connection


# =============================================================================
# SQL Execution Functions
# =============================================================================
def _validate_query(query: str) -> None:
    """Validate SQL query for safety (SELECT only)."""
    query_upper = query.strip().upper()
    # Block DDL/DML operations
    blocked_keywords = [
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
        "TRUNCATE", "GRANT", "REVOKE", "COPY", "ATTACH", "DETACH",
    ]
    for keyword in blocked_keywords:
        if query_upper.startswith(keyword) or f" {keyword} " in query_upper:
            raise ValueError(f"Operation '{keyword}' is not allowed. Only SELECT queries are permitted.")


def execute_delta_sql_duckdb(project_name: str, query: str, limit: int = SQL_DEFAULT_LIMIT) -> dict:
    """Execute SQL query against Delta Lake using DuckDB with retry on connection errors."""
    delta_path = DELTA_PATHS.get(project_name)
    if not delta_path:
        return {"error": f"Unknown project: {project_name}"}

    _validate_query(query)

    # Enforce limit
    limit = min(limit, SQL_MAX_LIMIT)

    # Replace 'data' table reference with delta_scan
    # Simple replacement - assumes table is named 'data'
    modified_query = query.replace("FROM data", f"FROM delta_scan('{delta_path}')")
    modified_query = modified_query.replace("from data", f"FROM delta_scan('{delta_path}')")

    # Add LIMIT if not present
    if "LIMIT" not in modified_query.upper():
        modified_query = f"{modified_query} LIMIT {limit}"

    # Retry logic: try once, if connection error retry with fresh connection
    for attempt in range(2):
        start_time = time.time()
        try:
            conn = _get_duckdb_connection(force_reconnect=(attempt > 0))
            result = conn.execute(modified_query).fetchdf()

            execution_time_ms = (time.time() - start_time) * 1000

            return {
                "columns": result.columns.tolist(),
                "data": result.to_dict(orient="records"),
                "row_count": len(result),
                "execution_time_ms": execution_time_ms,
                "engine": "duckdb",
            }
        except Exception as e:
            error_str = str(e).lower()
            # Retry on connection-related errors
            if attempt == 0 and ("connection" in error_str or "closed" in error_str or "invalid" in error_str):
                print(f"[SQL] Connection error, retrying with fresh connection: {e}")
                continue
            return {"error": str(e)}

    return {"error": "Failed after retry"}


def get_delta_table_schema(project_name: str) -> dict:
    """Get schema and metadata for a Delta Lake table using DuckDB with retry."""
    delta_path = DELTA_PATHS.get(project_name)
    if not delta_path:
        return {"error": f"Unknown project: {project_name}"}

    # Retry logic: try once, if connection error retry with fresh connection
    for attempt in range(2):
        try:
            conn = _get_duckdb_connection(force_reconnect=(attempt > 0))

            # Get schema using DESCRIBE
            schema_query = f"DESCRIBE SELECT * FROM delta_scan('{delta_path}')"
            schema_result = conn.execute(schema_query).fetchdf()

            columns = [
                {
                    "name": row["column_name"],
                    "type": row["column_type"],
                    "nullable": row.get("null", "YES") == "YES",
                }
                for _, row in schema_result.iterrows()
            ]

            # Get approximate row count
            try:
                count_query = f"SELECT COUNT(*) as cnt FROM delta_scan('{delta_path}')"
                count_result = conn.execute(count_query).fetchone()
                row_count = count_result[0] if count_result else 0
            except Exception:
                row_count = 0

            return {
                "table_name": project_name.lower().replace(" ", "_"),
                "delta_path": delta_path,
                "columns": columns,
                "approximate_row_count": row_count,
            }
        except Exception as e:
            error_str = str(e).lower()
            # Retry on connection-related errors
            if attempt == 0 and ("connection" in error_str or "closed" in error_str or "invalid" in error_str):
                print(f"[SQL] Schema query connection error, retrying: {e}")
                continue
            return {"error": str(e)}

    return {"error": "Failed after retry"}


# =============================================================================
# API Endpoints
# =============================================================================
@router.post("/query")
async def sql_query(request: SQLQueryRequest):
    """
    Execute a SQL query against Delta Lake using DuckDB.

    The query runs against the Delta Lake table for the specified project.
    The table is accessible as 'data' in your SQL queries.

    Example queries:
    - SELECT * FROM data LIMIT 100
    - SELECT COUNT(*) FROM data WHERE amount > 1000
    - SELECT merchant_id, AVG(amount) FROM data GROUP BY merchant_id

    Security:
    - Only SELECT queries are allowed
    - DDL/DML operations are blocked
    - Row limits are enforced
    """
    try:
        # Run the query in a thread pool using DuckDB
        result = await asyncio.wait_for(
            asyncio.to_thread(
                execute_delta_sql_duckdb,
                request.project_name,
                request.query,
                request.limit,
            ),
            timeout=SQL_QUERY_TIMEOUT,
        )

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Query timed out after {SQL_QUERY_TIMEOUT} seconds. Try a simpler query or add LIMIT.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@router.post("/schema")
async def table_schema(request: TableSchemaRequest):
    """
    Get schema and metadata for a Delta Lake table.

    Returns:
    - table_name: Name of the Delta Lake table
    - delta_path: S3 path to the Delta Lake table
    - columns: List of column definitions (name, type, nullable)
    - approximate_row_count: Approximate number of rows in the table
    """
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(get_delta_table_schema, request.project_name),
            timeout=30.0,
        )

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Schema query timed out after 30 seconds.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get table schema: {str(e)}")


@router.post("/total-rows")
async def get_total_rows(request: TableSchemaRequest):
    """
    Get total number of rows in a Delta Lake table.

    This is useful for setting the maximum value for training row limits.
    """
    try:
        delta_path = DELTA_PATHS.get(request.project_name)
        if not delta_path:
            raise HTTPException(status_code=400, detail=f"Unknown project: {request.project_name}")

        def query_count():
            conn = _get_duckdb_connection()
            result = conn.execute(f"SELECT COUNT(*) FROM delta_scan('{delta_path}')").fetchone()
            return result[0] if result else 0

        total_rows = await asyncio.wait_for(
            asyncio.to_thread(query_count),
            timeout=30.0,
        )

        return {"total_rows": total_rows, "project_name": request.project_name}

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Query timed out")
    except HTTPException:
        raise
    except Exception as e:
        return {"total_rows": 0, "project_name": request.project_name, "error": str(e)}
