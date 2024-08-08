class ZeroShotTemplate:
    @staticmethod
    def generate_answer(text):
        return f"""Based on the given open-ended questions, please generate the answer.

Quersion:
{text}

Answer:
"""