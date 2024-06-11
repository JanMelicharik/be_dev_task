from langchain_core.documents.base import Document


PROMPT_TEMPLATE = """
You have the following context:

{context}

And you have the following question: {question}

If you can find the answer to the question in the context, provide the answer and the source of the information
without content and be {verbosity}.
If you cannot find the answer to the question in the provided context, don't provide an answer and instead
provide the following key: {no_answer_key}
"""


def prompt_template(resources: list[Document], question: str, no_answer_key: str, verbose: bool) -> str:
    resource_texts: list[str] = list()

    for index, resource in enumerate(resources):
        resource_source = resource.metadata["source"]
        resource_texts.append(f'Source: {resource_source}\nContent: """{resource.page_content}"""')

    context = "\n\n---\n\n".join(resource_texts)
    verbosity = "verbose" if verbose else "succinct"

    return PROMPT_TEMPLATE.format(context=context, question=question, no_answer_key=no_answer_key, verbosity=verbosity)