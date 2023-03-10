import os
import matplotlib.pyplot as plt

def parse_line(line):
    datastring = line.split(':')[1]
    datastring = datastring[1:-2]
    data = datastring.split(', ')
    return list(map(float, data))

def extract_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    trainA = parse_line(lines[0])
    evalA = parse_line(lines[1])
    train_az = parse_line(lines[2])
    eval_az = parse_line(lines[3])
    return trainA, evalA, train_az, eval_az

def main():
    save_names = ['no_occlusion', 'default', 'no_occlusion_scale_variation', 'scale_variation']
    paper_names = ['default', 'occlusion', 'scale_variation', 'occlusion & scale variation']
    colors = ['r','g', 'b','k']
    fig, axs = plt.subplots(1,2)
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('average |A|')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('average a')
    for save_name, plot_name, c in zip(save_names, paper_names, colors):
        file = os.path.join('saved_a_values', '{}.txt'.format(save_name))
        _, eval_A, _, eval_az = extract_data(file)
        print(eval_A)
        axs[0].plot(range(len(eval_A)), eval_A,c, label=plot_name)
        axs[1].plot(range(len(eval_A)), eval_az,c, label=plot_name)
    axs[0].legend()
    plt.show()

if __name__ == '__main__':
    main()
