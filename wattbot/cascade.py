"""Use the multimodal solver when clean-index RAG abstains."""

from threading import Lock

from .multimodal import (
    FigureCatalog, MultimodalSearchIndex, VisionClient, detect_figure_pages,
    solve_multimodal,
)
from .pipeline import BLANK, COLUMNS, solve, validate_rows
from .retrieval import sha256_file


def visual_fallback(args, metadata):
    lock = Lock()
    resources = None

    def answer(question, emit):
        nonlocal resources
        with lock:
            if resources is None:
                index = MultimodalSearchIndex(args.visual_index)
                if set(index.sources) != set(metadata) or any(
                    index.sources[ref]["url"] != metadata[ref]["url"] for ref in metadata
                ):
                    raise ValueError("Visual index does not match the current metadata")
                for ref, source in index.sources.items():
                    path = args.docs / f"{ref}.pdf"
                    if not path.is_file() or sha256_file(path) != source["sha256"]:
                        raise ValueError(f"Visual index does not match {path}")
                catalog = FigureCatalog(detect_figure_pages(metadata, args.docs))
                client = VisionClient(args.vlm_model, args.vision_provider)
                resources = index, catalog, client
        index, catalog, client = resources
        return solve_multimodal(question, index, catalog, args.docs, client, emit,
                                top_k=args.top_k, mode="mode2-only", image_cache={})

    return answer


def solve_cascade(question, index, client, emit, top_k, metadata, fallback):
    def rag_event(event, **fields):
        emit(event, pipeline_stage="rag", **fields)

    row = solve(question, index, client, rag_event, top_k)
    validate_rows(COLUMNS, [row], [question], metadata)
    escalated = row["answer_value"] == BLANK
    emit("cascade_route", escalated=escalated,
         reason="rag_abstained" if escalated else "rag_answered")
    if not escalated:
        return row

    def visual_event(event, **fields):
        emit(event, pipeline_stage="multimodal", **fields)

    row = fallback(question, visual_event)
    validate_rows(COLUMNS, [row], [question], metadata)
    emit("cascade_result", selected="abstention" if row["answer_value"] == BLANK
         else "multimodal")
    return row
