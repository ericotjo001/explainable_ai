from src.settings.result_settings import HYPER_BOXPLOT_SETTINGS
import matplotlib.pyplot as plt
import joblib

def arrange_boxplots_of_hyper_results(dargs):
    print('arrange_boxplots_of_hyper_results')

    def add_boxplot(accs, text):
        kwidths = [k for k in accs]
        labels = [str(k) for k in kwidths]
        labels.insert(0,'')

        plt.gca().boxplot([y for x,y in accs.items()], 
            flierprops={'marker':'.', 
            'markeredgecolor':'k',
            'markerfacecolor':'k', 
            'markersize':4,}
        )
        
        plt.gca().set_xlim([0.5,None])
        plt.xticks(range(len(kwidths)+1), labels  )
        plt.gca().set_title(text)
        plt.gca().set_xlabel('k')


    plt.figure()
    plt.gcf().add_subplot(121)
    add_boxplot(joblib.load(HYPER_BOXPLOT_SETTINGS['mnist_hyper_dir']),'mnist')
    plt.gcf().add_subplot(122)
    add_boxplot(joblib.load(HYPER_BOXPLOT_SETTINGS['cifar_hyper_dir']),'CIFAR')
    plt.tight_layout()
    plt.show()