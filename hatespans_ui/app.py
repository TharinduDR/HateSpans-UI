from ast import literal_eval

import streamlit as st
from annotated_text import annotated_text
from hatespans.app.hate_spans_app import HateSpansApp
import pandas as pd


def toxic_to_rgb(is_toxic: bool):
    if is_toxic:
        return "rgb(255, 0, 0)"
    else:
        return "rgb(211,211,211)"


def highlight(s, index, color='yellow'):
    if s.name == index:
        hl = f"background-color: {color}"
    else:
        hl = ""
    return [hl] * len(s)


def get_data(dataset_name):
    if dataset_name == "Civil Comments Dataset":
        data = pd.read_csv("hatespans_ui/assets/data/tsd_trial.csv")
        data["spans"] = data.spans.apply(literal_eval)
        return data

    else:
        return None


def get_model(model_name):
    if model_name == "small":
        return HateSpansApp("small", use_cuda=False)

    if model_name == "large":
        return HateSpansApp("small", use_cuda=False)

    else:
        return None


# Keep the state between actions
@st.cache(allow_output_mutation=True)
def current_sentence_state():
    return {"index": 0}


def main():
    st.set_page_config(
        page_title='Hate Spans UI',
        initial_sidebar_state='expanded',
        layout='wide',
    )

    st.sidebar.title("Hate Spans")
    st.sidebar.markdown("Predict Hate Spans in your text")
    st.sidebar.markdown(
        "[code](https://github.com/TharinduDR/HateSpans)"
    )

    st.sidebar.markdown("---")

    st.sidebar.header("Available Datasets")
    selected_dataset_name = st.sidebar.radio(
        'Select a dataset to use',
        ["Civil Comments Dataset"]
    )

    df = get_data(selected_dataset_name)

    st.sidebar.markdown("---")

    st.sidebar.header("Available Models")
    selected_model = st.sidebar.radio(
        'Select a pretrained model to use',
        ["small", "large"],
    )

    model = get_model(selected_model)

    st.header("Input a sentence")
    st.write(
        "Select a predefined sentence and/or edit the sentences"
    )

    current_state = current_sentence_state()

    col1, col2, *_ = st.beta_columns(12)
    with col1:
        previous_pressed = st.button('Previous')
    with col2:
        next_pressed = st.button('Next')

    if previous_pressed:
        current_state['index'] = max(0, current_state['index'] - 1)
    if next_pressed:
        current_state['index'] = min(len(df), current_state['index'] + 1)

    i = st.slider(
        'Scroll through dataset',
        min_value=0,
        max_value=len(df['text'].tolist()),
        value=current_state['index'],
    )
    current_state['index'] = i

    sentences = df['text'].tolist()
    sentence = sentences[i]

    with st.beta_expander('Preview sentences', expanded=False):
        first_idx = max(i - 7, 0)
        last_idx = min(first_idx + 15, len(sentences))
        df_w = df.iloc[first_idx:last_idx].style.apply(highlight, index=i, axis=1)
        st.dataframe(df_w, width=None, height=400)

    st.write('### Edit sentence')
    col1, col2 = st.beta_columns(2)
    with col1:
        sentence_text = st.text_area('Sentence', value=sentence)

    st.header('Toxic Spans')
    tokens = model.predict_tokens(sentence_text)

    predictions = st.beta_container()
    with predictions:
        text = [
            (token.text, "", toxic_to_rgb(token.is_toxic))
            for token in tokens
            ]
        st.write('Predicted Toxic spans in the sentence')
        annotated_text(*text)


if __name__ == "__main__":
    main()




