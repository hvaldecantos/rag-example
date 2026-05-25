import os
from typing import Optional

from IPython.display import Image, display


def display_graph(abot_graph, filename: Optional[str] = None) -> None:

    try:
        graph_image = abot_graph.draw_mermaid_png()
        if filename:
            if os.path.exists(filename):
                print(f"File '{filename}' already exists. Skipping save.")
            else:
                with open(filename, 'wb') as f:
                    f.write(graph_image)
        display(Image(graph_image))
    except Exception as e:
        print("Error displaying graph:", e)
