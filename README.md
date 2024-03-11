# Information-Retrieval
## Overview
This project aims to build a search engine for the English Wikipedia corpus. Leveraging the entire Wikipedia dump and employing stemming techniques and removal of rare words, the search engine retrieves relevant results for user queries. It supports the BM25 ranking method and ensures efficient query processing within a specified time limit. The project also focuses on quality evaluation, with metrics such as Average Precision@10, and provides a functional search frontend accessible via URL for testing purposes.


## requirements
Python 3 version

Pandas

GCP (optional for deployment)

## Getting Started

In order to use this engine  upload the run_in_frontend to google colab.

The first step after uploading the notebook is to upload a query Json file(like the file called queries_train) and run the cell that imports Json.

The next step is to	 run the follwowing 3 cells.

Before running the engine enter http://34.70.143.109:8080 as your URL.

Run the last cell and change the metrics as desired.

## Contributors
Amit Cohen

Dvir Chitrit

