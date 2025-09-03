from fastapi import FastAPI
from pydantic import BaseModel
from agent.agentic_workflow import GraphBuilder
from fastapi.responses import JSONResponse
import os

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_travel_agent(query: QueryRequest):
    try:
        print(query)
        graph = GraphBuilder(model_provider="groq")
        react_app = graph()

        png_graph = react_app.get_graph().draw_mermaid_png()
        with open("my_graph.png", "wb") as f:
            f.write(png_graph)

        print(f"Graph image saved as my_graph.png in { os.getcwd() }")

        message = {"message": [query.question]}

        output = react_app.invoke(message)

        if isinstance(output, dict) and "message" in output:
            final_output = output["message"][-1].content
        else:
            final_output = str(output)

        return {"answer" : final_output}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

