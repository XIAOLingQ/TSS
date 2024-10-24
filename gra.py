from sup import *
import os

def build_demo(embed_mode, cur_dir=None, concurrency_count=10):

    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="TiAmo", theme=gr.themes.Default()) as demo:
        state = gr.State()
        print(state)
        if not embed_mode:
            gr.Markdown("# TiAmo Chatbot")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False
                    )

                # 将 interactive 设置为 False，禁止图片交互
                imagebox = gr.Image(type="pil", interactive=False)
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image",
                    visible=False
                )

                if cur_dir is None:
                    cur_dir = os.path.dirname(os.path.abspath(__file__))

                gr.Examples(examples=[
                    [f"1.jpeg", "How about this Tibetan costume."],
                    [f"2.jpg", "Describe the Miao costumes in the picture."]
                ], inputs=[imagebox, textbox])

                # 模型参数
                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                            label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P")
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True,
                                                  label="Max output tokens")

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="TiAmo Chatbot",
                    height=650,
                    layout="panel",
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")

                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="  Downvote", interactive=False)
                    flag_btn = gr.Button(value="  Flag", interactive=False)
                    regenerate_btn = gr.Button(value="  Regenerate", interactive=False)
                    clear_btn = gr.Button(value=" Clear", interactive=True)
                    generate_img_btn = gr.Button(value="Generate Image", variant="secondary")

        # Register listeners (移除涉及后端接口的监听器)
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn, generate_img_btn]

        clear_btn.click(
            lambda: (None, None, "", None) + (False,) * len(btn_list),
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list
        ).then(
            resllava,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
        )

        generate_img_btn.click(
            generate_image_from_last_response,
            [chatbot],
            [imagebox]
        )

    return demo

demo = build_demo(embed_mode=False)
demo.launch()
