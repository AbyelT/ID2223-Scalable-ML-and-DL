import gradio as gr
import hopsworks

project = hopsworks.login()

dataset_api = project.get_dataset_api()
dataset_api.download("Resources/images/df_se3_elec_price_recent.png", overwrite=True)
dataset_api.download("Resources/images/df_se3_error_metrics.png", overwrite=True)
dataset_api.download("Resources/images/df_se3_elec_price_prediction.png", overwrite=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Label("Today's prediction")
            input_img = gr.Image("df_se3_elec_price_recent.png", elem_id="latest-prediction")
            input_img = gr.Image("df_se3_error_metrics.png", elem_id="latest-error-metrics")
        with gr.Column():
            gr.Label("Prediction History")
            input_img = gr.Image("df_se3_elec_price_prediction.png", elem_id="historical-predictions")

demo.launch()