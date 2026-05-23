"""Wire the metalens Self-RAG graph from injected dependencies.

The graph nodes are:
    parse → retrieve → grade →(cond)→ generate_params → simulate_rcwa
        → optimize → evaluate → reflect →(cond)→ {finalize | retrieve |
                                                    simulate_rcwa | optimize}
        → END

Build via:
    graph = build_metalens_self_rag_graph(provider, kb, grader, tools)
    result = graph.invoke({"requirement": "...", "target_strehl": 0.9})
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from metaopticsai.llm.base import LLMProvider
from metaopticsai.rag.base import KnowledgeBase
from metaopticsai.rag.grader import DocumentGrader
from metaopticsai.tools.base import ToolRegistry
from metaopticsai.workflows.edges import route_after_grade, route_after_reflect
from metaopticsai.workflows.nodes import (
    make_parse_node, make_retrieve_node, make_grade_node,
    make_generate_node, make_simulate_node, make_optimize_node,
    make_evaluate_node, make_reflect_node, make_finalize_node,
)
from metaopticsai.workflows.state import AutoMLState


def build_metalens_self_rag_graph(
    provider: LLMProvider,
    kb: KnowledgeBase,
    grader: DocumentGrader,
    tools: ToolRegistry,
):
    g = StateGraph(AutoMLState)

    # Register nodes
    g.add_node("parse", make_parse_node(provider))
    g.add_node("retrieve", make_retrieve_node(kb))
    g.add_node("grade", make_grade_node(grader))
    g.add_node("generate_params", make_generate_node(provider))
    g.add_node("simulate_rcwa", make_simulate_node(tools))
    g.add_node("optimize", make_optimize_node(tools))
    g.add_node("evaluate", make_evaluate_node(tools))
    g.add_node("reflect", make_reflect_node(provider))
    g.add_node("finalize", make_finalize_node(tools, provider))

    # Linear edges
    g.add_edge(START, "parse")
    g.add_edge("parse", "retrieve")
    g.add_edge("retrieve", "grade")
    # Conditional after grade
    g.add_conditional_edges("grade", route_after_grade, {
        "retrieve": "retrieve",
        "generate_params": "generate_params",
    })
    g.add_edge("generate_params", "simulate_rcwa")
    g.add_edge("simulate_rcwa", "optimize")
    g.add_edge("optimize", "evaluate")
    g.add_edge("evaluate", "reflect")
    # Conditional after reflect
    g.add_conditional_edges("reflect", route_after_reflect, {
        "retrieve": "retrieve",
        "simulate_rcwa": "simulate_rcwa",
        "optimize": "optimize",
        "finalize": "finalize",
    })
    g.add_edge("finalize", END)

    return g.compile()