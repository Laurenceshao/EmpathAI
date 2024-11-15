This project is based on [CHA](https://github.com/Institute4FutureHealth/CHA), originally developed by the Institute4FutureHealth. 

We have made modifications to adapt it for our own use case:

suicidal_sensor.py: use the fine-tuned text classification model based on distilbert-base-uncased to label user prompt

orchestrator.py: modify the current workflow to first incorporate suicide-risk classification given a user query, generate follow-up questions based on the detected risk-level, and gather user responses before proceeding with the normal planning

follow_up_generator.py: added an additional follow-up-generator component to come up with three follow-up questions given the user assessed suicide risk-level 

vector_database.py: implemented vector database framework for future RAG-enhancements, ex/ generating synthetic data based on “supportive” labeled messages for the LLM to reference for empathetic conversations (currently implemented with few-shot prompting) 


our huggingface classifer model output (checkpoiunt, eval_result, tokenizers, etc.): 

https://huggingface.co/Butsushi/results

https://huggingface.co/Butsushi/suicidal_sensor_class_weighted
