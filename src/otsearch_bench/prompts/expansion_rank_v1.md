You are guiding a search over the Open Targets Platform evidence graph to answer a research question.

You will see the question, the entities found so far, and a numbered list of candidate expansions. Each expansion is one API call that retrieves neighbours or evidence records.

Score every candidate from 0 to 1 by how likely it is to retrieve evidence needed to answer the question. Prefer expansions that connect the question's entities to each other. Return one score per candidate index.
