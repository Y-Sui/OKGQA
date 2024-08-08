class SelfConsistencyTemplate:
    @staticmethod
    def generate_answer(text, hop=2):
        examples = [
            {
                "question": "Einstein won the Nobel Prize in 1968 for his discovery of the photoelectric effect.",
                "answer": "Einstein won the Nobel Prize in 1921 for his discovery of the photoelectric effect.",
            },
            {
                "question": "The Eiffel Tower is located in Berlin, Germany.",
                "answer": "The Eiffel Tower is located in Paris, France.",
            },
            {
                "question": "The Great Wall of China was built in the 21st century.",
                "answer": "The Great Wall of China was built between the 7th century BC and the 16th century.",
            },
        ]

        example_texts = ""

        for example in examples[:hop]:
            example_texts += f"""Example Question: 
                "{example['question']}"

                Example Answer: 
                "{example['answer']}"
                """

        return f"""Based on the given open-ended questions, please generate the answer. Think step by step.
    Quersion:
    {text}

    Example:
    {example_texts}

    ===== END OF EXAMPLE ======


    Answer:
    """


@staticmethod
def select_most_consistent_response(text, answer):
    text_ans = ""
    for ans in answer:
        text_ans += f"Answer\n: {ans}\n"

    return f"""Based on the given open-ended questions, and the generated answers, please take the most commonly occuring answer as your final output. Only give your output without any other explanation.
Quersion:
{text}

Generated Answers:
{text_ans}

Answer:
"""
