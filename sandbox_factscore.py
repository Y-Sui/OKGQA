"""
@inproceedings{ factscore,
    title={ {FActScore}: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation },
    author={ Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2023 },
    booktitle = { EMNLP },
    url={ https://arxiv.org/abs/2305.14251 }
}
"""

import os
from dotenv import load_dotenv

load_dotenv()

# # generations = a list of texts you want to calculate FactScore for
# # knowledge_sources = a list of texts that the generations were created from

# scores, init_scores = FactScore.get_factscore(generations, knowledge_sources)


# # {
# #     "input": "Question: Tell me a bio of Chief Jones.",
# #     "output": "I'm sorry, but I cannot provide information on Chief Jones as there are likely many individuals with that name. Could you please provide more specific information or context to help me identify which Chief Jones you are referring to?",
# #     "topic": "Chief Jones",
# #     "cat": ["very rare", "North America"],
# #     "annotations": null,
# # }


from factscore.factscorer import FactScorer

fs = FactScorer(openai_key="api.key")

out = fs.get_score(
    ["Jeff Hammond"],
    [
        "I'm sorry, but I cannot provide information on Chief Jones as there are likely many individuals with that name. Could you please provide more specific information or context to help me identify which Chief Jones you are referring to?"
    ],
)


print(out)


# import sqlite3

# # Create a SQL connection to our SQLite database
# con = sqlite3.connect("/data/yuansui/kg/.cache/factscore/enwiki-20230401.db")
# cur = con.cursor()

# # # query the database to get the table names
# # res = cur.execute("SELECT name FROM sqlite_master;")

# res = cur.execute("SELECT * FROM documents")
# all_data = cur.fetchall()
# print("\nTable data:")
# for row in all_data[:5]:
#     print(row)
