# ThenNothingLLM
Working with deepseek to make a neural net/LLM version of the never ending text project. Eventual goal is for it to be web-based and to read out the tetx. I think an interface should allow user to determine length of reading, and probably other parameters, style, author etc.
Deepseek is going for a hybrid approach:
  Fine-tune a transformer model to classify descriptive vs. non-descriptive text
  Use pre-trained embeddings for semantic similarity with known descriptive passages
  Implement a voting system combining multiple approaches
