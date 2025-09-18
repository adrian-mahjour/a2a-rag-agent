from langgraph.graph.state import CompiledStateGraph


def stream_graph_updates(graph: CompiledStateGraph, user_input: str, config: dict):

    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]}, config=config
    ):
        for node, update in event.items():
            print("Update from node: ", node)
            update["messages"][-1].pretty_print()
            print("\n\n")
