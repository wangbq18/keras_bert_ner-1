from Model.BertBilstmCrf import bert_bilstm_crf
from Model import DealWithData




if __name__ == "__main__":
    #数据
    train_path = "./Data/train1.txt"
    test_path = "./Data/test.txt"
    train_data = DealWithData.PreProcessData(train_path)
    test_data = DealWithData.PreProcessData(test_path)

    #模型
    max_seq_length = 80
    batch_size = 24
    epochs = 2
    lstmDim = 64
    model = bert_bilstm_crf(max_seq_length, batch_size, epochs, lstmDim)
    #model.TrainModel(train_data, test_data)


    #测试
    while 1:
        sentence = input('please input sentence:\n')
        tag = model.ModelPredict(sentence)
        print(tag)
