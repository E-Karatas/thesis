# Library ml
if __name__ == "__main__":
    print("This is library ""ml"" with functions related to importing and processing data for machine learning.")
elif __name__ == "sample.ml":
    print("Loaded library ml")


def read_excel(file_name, sheet_name=0):
    # Reading excel file and output PANDAS dataframe
    import pandas
    xls = pandas.ExcelFile(file_name)
    df = xls.parse(xls.sheet_names[sheet_name])
    df.to_dict()
    return df


def make_composition(df):
    # Identify element columns in dataframe and create composition object from it
    from pymatgen.core.structure import Composition, Element
    import numpy as np
    number_of_entries = df[df.columns[0]].size
    c = []
    for i in range(number_of_entries):
        elements = dict()
        for el in Element:
            if any(el.name == df.columns):
                ind = (np.where(el.name == df.columns)[0])
                value = df[df.columns[ind]].values[i]
                elements.update({el: value[0]})
                elements[el] = elements[el].astype(float)
                # elements.append(el.name)
        if all(np.isnan(val) for val in elements.values()):
            c.append(None)
        else:
            c.append(Composition(elements))
    # Delete element columns in df
    #for key in elements.keys():
        #del df[key.name]
    df['composition'] = c
    return df


def convert_composition(composition, convert_to='atomic'):
    import numpy as np
    from pymatgen.core.structure import Composition
    k = 0
    for c in composition.composition:
        new_composition = dict()
        if abs(c.num_atoms - 1) > 1e-10:
            raise NameError("Composition %s does not add up to 100%%" % c.formula)
        element_name = [e.name for e in c.elements]
        atomic_mass = [e.atomic_mass for e in c.elements]
        element_fraction = [c[e] for e in c.elements]
        if convert_to == 'weight':
            weight = np.multiply(element_fraction, atomic_mass)
            weight_fraction = weight / np.sum(weight)
            fraction = weight_fraction
        elif convert_to == 'atomic':
            nr_atoms = np.divide(element_fraction, atomic_mass)
            atomic_fraction = nr_atoms / (np.sum(nr_atoms))
            fraction = atomic_fraction
        for el, frac in zip(element_name, fraction):
            new_composition.update({el: frac})
        composition.composition[k] = Composition(new_composition)
        k += 1
    return composition


def plot_measured_vs_predicted(y_measured, y_predicted, set_name=''):
    # Plot measured vs predicted target value
    import matplotlib.pyplot as plt
    plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    plt.scatter(y_measured, y_predicted, facecolors='none', edgecolors='r')
    plt.plot([y_measured.min(), y_measured.max()], [y_measured.min(), y_measured.max()], 'k--', lw=4)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title(set_name)
    plt.figaspect(1.)
    plt.show()


# Plot bar diagram showing the importance of features in Random forests method
def plot_feature_importance(X, est):
    # Tree's Feature Importance from Mean Decrease in Impurity (MDI)
    import matplotlib.pyplot as plt
    import numpy as np

    # ************** Alternative plot **************
    # # Gather data
    # objects = X.columns
    # y_pos = np.arange(len(objects))
    # feature_importance = 100. * est.feature_importances_
    # # P lot bar diagram
    # plt.figure(num=None, figsize=(6, 8), dpi=200, facecolor='w', edgecolor='k')
    # plt.bar(y_pos, feature_importance, align='center', alpha=0.5, color='red')
    # plt.xticks(y_pos, objects, rotation='vertical')
    # plt.ylabel('Feature Importance (%)')
    # plt.title('RandomForests - Feature Importance')
    # plt.show

    y_ticks = np.arange(0, len(X.columns))
    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_idx = est.feature_importances_.argsort()
    ax.barh(y_ticks, est.feature_importances_[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(X.columns[sorted_idx])
    ax.set_title("Random Forest Feature Importances (MDI)")
    fig.tight_layout()
    plt.show()


def plot_permutation_importance(est, X, y, set_name=''):
    from sklearn.inspection import permutation_importance
    import matplotlib.pyplot as plt
    result = permutation_importance(est, X, y, n_repeats=10,
                                    random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X.columns[sorted_idx])
    ax.set_title("Permutation Importances (%s)" % set_name)
    fig.tight_layout()
    plt.show()
