import numbers


def class_counts(rows, labels):
    """
    Counts the number of each type of example in a dataset.
    :param rows: array of samples
    :param labels: rows data labels.
    :return: a dictionary of label -> count.
    """
    counts = {cls: 0 for cls in set(labels)}  # a dictionary of label -> count.
    for idx, x in enumerate(rows):
        # in our dataset format, the label is always the last column
        label = labels[idx]
        counts[label] += 1
    return counts


def is_numeric(value):
    """
    Test if a value is numeric.
    :param value: input value
    :return: boolean value.
    """
    return isinstance(value, numbers.Number)


def unique_vals(rows, col):
    """
    Find the unique values for a column in a dataset.
    :param rows:
    :param col:
    :return:
    """
    """
    Find the unique values for a column in a dataset.
    """
    return set([row[col] for row in rows])


class Question:
    """
    A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question.
    """

    def __init__(self, column, column_idx, value):
        self.column = column
        self.column_idx = column_idx
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column_idx]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        val_str = str(self.value)
        if is_numeric(self.value):
            condition = ">="
            val_str = '{:.2f}'.format(self.value)
        return "%s \n %s" % (self.column, val_str)


class Leaf:
    """
    A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows, labels):
        self.predictions = class_counts(rows, labels)

    def __str__(self) -> str:
        """
        “informal” or nicely printable string representation of an object
        """
        pred = self.predictions
        pred_str = ''.join(f'{c} : {pred[c]} \n' for c in pred.keys())
        return pred_str


class DecisionNode:
    """
    A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __str__(self) -> str:
        """
        “informal” or nicely printable string representation of an object
        """
        return str(self.question)
