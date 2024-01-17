from ray.serve.gradio_integrations import GradioServer 

import gradio as gr 

def gradio_app_builder():

    def model(input_str: str) -> str:
        return input_str[::-1]
    
    return gr.Interface(
        fn=lambda x: model(x),
        inputs=gr.Textbox(label="Input text"),
        outputs=gr.Textbox(label="Reversed text")
    )

app = GradioServer.options(ray_actor_options={"num_cpus":1}).bind(gradio_app_builder)
