from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
data = ["I love you", "I hate you"]
sentiment_pipeline(data)


# from google.cloud import language_v1
# import six
#
#
# def sample_analyze_sentiment(content):
#
#     client = language_v1.LanguageServiceClient()
#
#     content = 'Your text to analyze, e.g. Hello, world!'
#
#     if isinstance(content, six.binary_type):
#         content = content.decode("utf-8")
#
#     type_ = language_v1.Document.Type.PLAIN_TEXT
#     document = {"type_": type_, "content": content}
#
#     response = client.analyze_sentiment(request={"document": document})
#     sentiment = response.document_sentiment
#     print("Score: {}".format(sentiment.score))
#     print("Magnitude: {}".format(sentiment.magnitude))