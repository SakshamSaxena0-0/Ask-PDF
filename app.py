import gradio as gr
from utils.pdf_parser import extract_text_from_pdf
from utils.rag_chain import RAGChatbot


chatbot = RAGChatbot()

def process_query(pdf, query, history):
    if pdf is not None:
        text = extract_text_from_pdf(pdf.name)
        chatbot.load_document(text)
    answer = chatbot.query(query)
    history.append((query, answer))
    return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 RAG‑based PDF Reader with Contextual Memory")
    pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
    chatbot_ui = gr.Chatbot()
    user_input = gr.Textbox(label="Your Question")
    ask_button = gr.Button("Ask")

    ask_button.click(process_query, [pdf_file, user_input, chatbot_ui], [chatbot_ui, user_input])

if __name__ == "__main__":
    demo.launch()
