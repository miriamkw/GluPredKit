from src.plots.base_plot import BasePlot
import matplotlib.pyplot as plt


class CompareScatterPlot(BasePlot):

    def __call__(self, predictions, model_names, y_true, use_mgdl=True, title=''):
        """
        A plot of a prediction, where you can change the carbohydrate and insulin input with a slider, and observe
        how the model reacts.

        predictions -- a list of lists of predictions, one list per model
        model_names -- a list of model_names corresponding to the number of lists inside predictions
        df -- a dataframe that the prediction model can use as input to get a prediction
        use_mgdl -- whether to present results in mg/dL or mmol/L
        """
        if use_mgdl:
            k = 1
            unit_label = 'mg/dL'
        else:
            k = 18.0182
            unit_label = 'mmol/L'

        font = {
            'size': 18,
        }

        plt.rc('font', **font)

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25)

        x = [30 / k, 400 / k]
        y = [30 / k, 400 / k]

        ax.plot(x, y, color='black', linestyle='--', label='y = x')

        y_true = [el / k for el in y_true]

        for i in range(len(predictions)):
            y_pred = predictions[i]

            ax.scatter(y_true, [el / k for el in y_pred], label=model_names[i], alpha=0.5)

        #plt.legend(loc='upper left')

        plt.xlabel('True Blood Glucose [' + unit_label + ']')
        plt.ylabel('Predicted Blood Glucose [' + unit_label + ']')

        plt.title(title)

        plt.show()