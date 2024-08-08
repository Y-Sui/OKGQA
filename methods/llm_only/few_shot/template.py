class FewShotTemplate:
    @staticmethod
    def generate_answer(text, hop=2):
      
      examples = [
            {
                "question": "Einstein won the Nobel Prize in 1968 for his discovery of the photoelectric effect.",
                "answer": "Einstein won the Nobel Prize in 1921 for his discovery of the photoelectric effect."
            },
            {
                "question": "The Eiffel Tower is located in Berlin, Germany.",
                "answer": "The Eiffel Tower is located in Paris, France."
            },
            {
                "question": "The Great Wall of China was built in the 21st century.",
                "answer": "The Great Wall of China was built between the 7th century BC and the 16th century."
            }
        ]
      
      example_texts = ""
      for example in examples[:hop]:
         example_texts += f"""Example Question: 
            "{example['question']}"

            Example Answer: 
            "{example['answer']}"
            """

      return f"""Based on the given open-ended questions, please generate the answer.
Quersion:
{text}

Example:
{example_texts}

===== END OF EXAMPLE ======


Answer:
"""