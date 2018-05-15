# %matplotlib inline
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import jpype
import os
from tqdm import tqdm

# 分词java程序jar包
jarpath = r'E:/Python/jupyter-notebook/IKAnalyzer.jar'
# 打开java虚拟机
jpype.startJVM(jpype.getDefaultJVMPath(),"-ea", "-Djava.class.path=%s" % jarpath)
# 加载类并实例化
IkAnalyzer = jpype.JClass('com.IkAnalyzer')
ana= IkAnalyzer()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Intelligent_QAS():
    # 词向量文件
    glove_vectors_file = "glove.6B.50d.txt"

    # 训练和测试文件
    train_file_name = "train.txt"
    test_file_name = "test.txt"
    # train_file_path = "tasks/" + train_file_name
    train_file_path = "tasks_1-20_v1-2/en/" + train_file_name
    # test_file_path = "tasks/" + test_file_name
    test_file_path = "tasks_1-20_v1-2/en/" + test_file_name

    def __init__(self):
        # 使用tqdm
        tqdm.monitor_interval = 0
        self.deserialize_GloVe_vectors()
        self.final_train_data = self.process_data_file(self.train_file_path)
        self.final_test_data = self.process_data_file(self.test_file_path)
        tf.reset_default_graph()
        self.init_hyperparameters()
        self.init_Input_module()
        self.init_Question_module()
        self.init_weights_and_bias()
        self.init_Episodic_Memory_module()
        self.init_Answer_module()
        self.optimizer_strategy()
        self.init_session()
        self.init_batch_data()

    # 反序列化词向量
    def deserialize_GloVe_vectors(self):
        self.glove_wordmap = {}
        with open(self.glove_vectors_file, "r", encoding="utf8")as gloves:
            for glove in gloves:
                word, vector = tuple(glove.split(" ", 1))
                self.glove_wordmap[word] = np.fromstring(vector, sep=" ")
        wvecs = []
        for item in self.glove_wordmap.items():
            wvecs.append(item[1])
        s = np.vstack(wvecs)

        # 获取分布超参数
        # 压缩行，对各列求协方差
        self.v = np.var(s, 0)
        # 压缩行，对各列求平均值
        self.m = np.mean(s, 0)
        # 局部随机化种子
        self.RS = np.random.RandomState()

    # 处理未知词汇
    def fill_unknow_word(self, unk):
        # 依照原有的均值和协方差随机生成词向量
        self.glove_wordmap[unk] = self.RS.multivariate_normal(self.m, np.diag(self.v))
        return self.glove_wordmap[unk]

    # 句子转化为序列
    def sentence2sequence(self, sentence):
        # 将输入的句子转化为一个(n,d)的矩阵，其中n是句子中词的数量，d是每个词向量具有的维数
        word_blocks = sentence.strip('"(),-').lower().split(" ")
        vectors = []
        words = []
        # 采用词尽量长的贪心方式获取词
        for word_block in word_blocks:
            wb_len = len(word_block)
            while len(word_block) > 0:
                temp_word = word_block[:wb_len]
                if temp_word in self.glove_wordmap:
                    vectors.append(self.glove_wordmap[temp_word])
                    words.append(temp_word)
                    word_block = word_block[wb_len:]
                    wb_len = len(word_block)
                    continue
                else:
                    wb_len = wb_len-1
                if wb_len == 0:
                    # 未知的词
                    vectors.append(self.fill_unknow_word(word_block))
                    words.append(word_block)
                    break
        return np.array(vectors), words

    # 上下文化问答数据
    def contextualize(self, file_name):
        """
        读取提问的数据集并构建“问答->上下文集”这样的关系.
        输出为一个七元组：[句子的向量形式, 句子的词构造, 问题的向量形式, 问题的词构造, 答案的向量形式, 答案的词构造, 一系列编号]
        """
        data = []
        qa_context = []
        with open(file_name, "r") as data_set:
            for line in data_set:
                # 将行号从行中分割出来
                line_number, line_res = tuple(line.split(" ", 1))

                if line_number == "1":
                    # 行号为1表示新的问题情景
                    qa_context = []
                if "\t" in line_res:
                    # Tab键作为问题、答案、编号的划分
                    question, answer, support = tuple(line_res.split("\t"))
                    data.append((tuple(zip(*qa_context)) +
                                self.sentence2sequence(question) +
                                self.sentence2sequence(answer) +
                                ([int(s) for s in support.split()],)))
                else:
                    # 句子的一部分
                    qa_context.append(self.sentence2sequence(line_res[:-1]))
        return data

    # 对上下文化过程中获得的数据进行最终处理
    def finalize(self, data):

        final_data = []
        for cqas in data:
            context_vectors, context_words, question_vectors, question_words, answer_vectors, answer_words, spt = cqas
            lengths = itertools.accumulate(len(context_vector) for context_vector in context_vectors)
            context_vec = np.concatenate(context_vectors)
            sum_context_words = sum(context_words, [])
            # 在新句子开始处标记
            sentence_ends = np.array(list(lengths))
            final_data.append((context_vec, sentence_ends, question_vectors, spt, sum_context_words, cqas, answer_vectors, answer_words))
        return np.array(final_data)

    def process_data_file(self, file_name):
        data = self.contextualize(file_name)
        final_data = self.finalize(data)
        return final_data

    # 初始化超参数
    def init_hyperparameters(self):
        # 存储网络中循环图层之间传递的数据的维数。
        self.recurrent_cell_size = 128

        # 词向量的维数
        self.D = 50

        # 神经网络学习的速度
        self.learning_rate = 0.005

        # dropout概率
        self.input_p = 0.5
        self.output_p = 0.5

        # 同时训练的问答数目
        self.batch_size = 128

        # 情景记忆的传递次数
        self.passes = 4

        # 前馈图层大小：存储前馈图层传来的数据的维度大小
        self.ff_hidden_size = 256

        # 规范化问答的权值，增加情景记忆的稀疏性，但会使训练变慢
        self.weight_decay = 0.00000001

        # 每次网络训练时问答的数目
        self.training_iterations_count = 400000

        # 每多少次迭代进行一次检查
        self.display_step = 100

    def init_Input_module(self):
        # context_placeholder: 包含上下文信息的张量，结构为:[批量大小,上下文信息最大长度,词向量维数]
        self.context_placeholder = tf.placeholder(tf.float32, [None, None, self.D], "context")

        # input_sentence_endings: 包含句子结束位置信息的张量，结构为 [批量大小,最大句子数,2]
        self.input_sentence_endings = tf.placeholder(tf.int32, [None, None, 2], "sentence")

        # recurrent_cell_size: 循环层的隐藏单元数
        self.input_gru = tf.contrib.rnn.GRUCell(self.recurrent_cell_size)

        self.gru_drop = tf.contrib.rnn.DropoutWrapper(self.input_gru, self.input_p, self.output_p)

        # dynamic_rnn 返回输出和对应的最终状态
        self.input_module_outputs, _ = tf.nn.dynamic_rnn(self.gru_drop,self.context_placeholder, dtype=tf.float32, scope="input_module")

        # context_facts: 从输出中通过索引获取事实
        self.context_facts = tf.gather_nd(self.input_module_outputs,self.input_sentence_endings)
        # facts: 将每个词都作为事实
        self.facts = self.input_module_outputs

    def init_Question_module(self):
        # question: 包含所有问题的张量，结构为[批量大小,问题最大长度,词向量维数]
        self.question = tf.placeholder(tf.float32, [None, None, self.D], "query")

        # input_question_lengths: 包含问题长度信息的张量，结构为[批量大小,2]
        # input_question_lengths[:,1]为实际长度;input_question_lengths[:,0] 为范围
        self.input_question_lengths = tf.placeholder(tf.int32, [None, 2], "query_lengths")

        self.question_module_outputs, _ = tf.nn.dynamic_rnn(self.gru_drop, self.question, dtype=tf.float32,
                                                            scope=tf.VariableScope(True, "input_module"))

        # ques: 问题状态张量，结构为[batch_size,recurrent_cell_size]
        self.ques = tf.gather_nd(self.question_module_outputs, self.input_question_lengths)

    def init_weights_and_bias(self):
        # 确保当前的记忆(即问题向量）沿着事实维度广播
        self.size = tf.stack([tf.constant(1), tf.shape(self.context_facts)[1], tf.constant(1)])
        self.re_ques = tf.tile(tf.reshape(self.ques, [-1, 1, self.recurrent_cell_size]), self.size)

        # 创建掩模需要注意力的最后输出为1
        output_size = 1

        # 权重和偏差
        attend_init = tf.random_normal_initializer(stddev=0.1)  # 生成一个正态分布的张量的初始化器
        self.weights_1 = tf.get_variable("attend_weights1", [1, self.recurrent_cell_size * 7, self.recurrent_cell_size],
                                    tf.float32, initializer=attend_init)
        self.weights_2 = tf.get_variable("attend_weights2", [1, self.recurrent_cell_size, output_size], tf.float32,
                                    initializer=attend_init)
        self.bias_1 = tf.get_variable("attend_bias1", [1, self.recurrent_cell_size], tf.float32, initializer=attend_init)
        self.bias_2 = tf.get_variable("attend_bias2", [1, output_size], tf.float32, initializer=attend_init)

        # 调整所有的权重和偏差
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.weights_1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.bias_1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.weights_2))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.bias_2))

    def attention(self, facts, memory, existing_facts):
        """
        自定义关注机制
        facts: 包含所有从上下文中获取的事实的张量，结构为 [批量大小,最大句子数,循环层隐藏单元数]
        memory: 包含当前记忆的张量，结构为 [批量大小,最大句子数,循环层隐藏单元数]
        existing_facts: 作为事实存在或不存在的二元掩模的张量，结构为 [批量大小,最大句子数,1]
        """
        with tf.variable_scope("attending") as scope:
            # attending: 我们决定注意的指标
            attending = tf.concat([facts, memory, self.re_ques, facts * self.re_ques, facts * memory,
                                   (facts - self.re_ques) ** 2, (facts - memory) ** 2], 2)

            # multiplied_weights_1: 前馈网络的第一层相乘权重
            multiplied_weights_1 = tf.matmul(attending * existing_facts,
                                             tf.tile(self.weights_1, tf.stack([tf.shape(attending)[0], 1, 1]))) * existing_facts

            # masked_bias_1: 仅针对存在事实的第一个前馈层偏差的掩藏版本
            masked_bias_1 = self.bias_1 * existing_facts

            # tnhan: 非线性激活函数
            tnhan = tf.nn.relu(multiplied_weights_1 + masked_bias_1)

            # multiplied_weights_2: 前馈网络的第二层相乘权重
            multiplied_weights_2 = tf.matmul(tnhan, tf.tile(self.weights_2, tf.stack([tf.shape(attending)[0], 1, 1])))

            # masked_bias_2: 第二个前馈层偏差的掩藏版本
            masked_bias_2 = self.bias_2 * existing_facts

            # norm_m2: 第二层权重的标准化版本
            norm_m2 = tf.nn.l2_normalize(multiplied_weights_2 + masked_bias_2, -1)

            # softmaxable: 使sparse_softmax方法能够在稠密张量上使用的一种转变
            softmax_idx = tf.where(tf.not_equal(norm_m2, 0))[:, :-1]
            softmax_gather = tf.gather_nd(norm_m2[..., 0], softmax_idx)
            softmax_shape = tf.shape(norm_m2, out_type=tf.int64)[:-1]
            softmaxable = tf.SparseTensor(softmax_idx, softmax_gather, softmax_shape)
            return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_softmax(softmaxable)), -1)

    def init_Episodic_Memory_module(self):
        # facts_Os: 表示相应事实是否存在的张量，存在为1，不存在为0，结构为A [batch_size,max_facts_length,1]
        self.facts_Os = tf.cast(tf.count_nonzero(self.input_sentence_endings[:, :, -1:], -1, keepdims=True), tf.float32)

        with tf.variable_scope("Episodes") as scope:
            attention_gru = tf.contrib.rnn.GRUCell(self.recurrent_cell_size)

            # memory: 一个注意力机制的记忆状态张量的列表
            self.memory = [self.ques]

            # attend_list: 网络关注的所有的张量的列表
            self.attend_list = []
            for i in range(self.passes):
                # 注意力掩码
                attend_to = self.attention(self.context_facts, tf.tile(tf.reshape(self.memory[-1], [-1, 1, self.recurrent_cell_size]), self.size), self.facts_Os)

                # 反转注意力掩码
                retain = 1 - attend_to

                # 通过注意力掩码传递事实
                while_valid_index = (lambda state, index: index < tf.shape(self.context_facts)[1])
                update_sate = (lambda state, index: (attend_to[:, index, :] * attention_gru(self.context_facts[:, index, :], state)[0] +
                                                     retain[:, index, :] * state))

                # 使用最近的记忆和第一个索引启动循环
                self.memory.append(tuple(tf.while_loop(while_valid_index,
                                                  (lambda state, index: (update_sate(state, index), index + 1)),
                                                  loop_vars=[self.memory[-1], 0]))[0])
                self.attend_list.append(attend_to)

                # 重用变量，使每次GRU传递使用相同的变量
                scope.reuse_variables()

    def init_Answer_module(self):
        # Answer Module

        # a0: 最终记忆状态（输入到答案模块中）
        a0 = tf.concat([self.memory[-1], self.ques], -1)

        # fc_init: 最终全连接层的权重的初始化器
        fc_init = tf.random_normal_initializer(stddev=0.1)

        with tf.variable_scope("answer"):
            # w_answer: 最终全连接层的权重
            w_answer = tf.get_variable("weight", [self.recurrent_cell_size * 2, self.D], tf.float32, initializer=fc_init)

            # 调整最终全连接层的权重
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_answer))

            # 回退词，并非实际的词，会对最相近的词进行匹配
            self.logit = tf.expand_dims(tf.matmul(a0, w_answer), 1)

            # 掩盖词的存在
            with tf.variable_scope("ending"):
                all_ends = tf.reshape(self.input_sentence_endings, [-1, 2])
                range_ends = tf.range(tf.shape(all_ends)[0])
                ends_indices = tf.stack([all_ends[:, 0], range_ends], axis=1)
                ind = tf.reduce_max(tf.scatter_nd(ends_indices, all_ends[:, 1], [tf.shape(self.ques)[0], tf.shape(all_ends)[0]]),
                                    axis=-1)
                range_ind = tf.range(tf.shape(ind)[0])
                mask_ends = tf.cast(tf.scatter_nd(tf.stack([ind, range_ind], axis=1), tf.ones_like(range_ind),
                                                  [tf.reduce_max(ind) + 1, tf.shape(ind)[0]]), bool)
                mask = tf.scan(tf.logical_xor, mask_ends, tf.ones_like(range_ind, dtype=bool))

            # 将所有可能的词与回退词进行欧几里得距离反比得分,得分最高的词将作为最后选择的词
            self.logit_score = -tf.reduce_sum(tf.square(self.context_placeholder * tf.transpose(tf.expand_dims(
                tf.cast(mask, tf.float32), -1), [1, 0, 2]) - self.logit), axis=-1)

    def optimizer_strategy(self):
        # gold_standard: 正确答案
        self.gold_standard = tf.placeholder(tf.float32, [None, 1, self.D], "answer")
        with tf.variable_scope('accuracy'):
            eq = tf.equal(self.context_placeholder, self.gold_standard)
            # 求与
            self.corrbool = tf.reduce_all(eq, -1)
            logloc = tf.reduce_max(self.logit_score, -1, keepdims=True)
            # logit_bool: 表示分数与最低分匹配与否的布尔型张量
            self.logit_bool = tf.equal(self.logit_score, logloc)

            # correctsbool: 表示上下文中哪个词总与最低分匹配
            correctsbool = tf.reduce_any(tf.logical_and(self.logit_bool, self.corrbool), -1)

            # corrects_float: correctsbool的浮点型张量
            self.corrects_float = tf.where(correctsbool, tf.ones_like(correctsbool, dtype=tf.float32),
                                tf.zeros_like(correctsbool, dtype=tf.float32))

            # corrects_answer: 类似corrects_float，但以正确答案取代我们选择的答案
            self.corrects_answer = tf.where(self.corrbool, tf.ones_like(self.corrbool, dtype=tf.float32),
                            tf.zeros_like(self.corrbool, dtype=tf.float32))

        with tf.variable_scope("loss"):
            # 以sigmoid交叉熵作为基本损失，距离作为相对概率，并标记属于答案的词在上下文中的位置
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.nn.l2_normalize(self.logit_score, -1), labels=self.corrects_answer)

            # 增加规范化损失，及其权重
            self.total_loss = tf.reduce_mean(loss) + self.weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # TensorFlow的Adam优化器默认实现工作
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # 尽量减少损失来适应训练
        self.optimizer_op = optimizer.minimize(self.total_loss)

    def init_session(self):
        # 创建saver对象，它添加了一些op用来save和restore模型参数
        self.saver = tf.train.Saver()

        # 初始化变量
        init = tf.global_variables_initializer()

        # 开启TensorFlow session
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver.restore(self.sess,"model/intelligent_qas.ckpt")

    def prep_batch(self, batch_data, more_data=False):
        """
        准备所有需要在批处理基础上完成的预处理
        """
        context_vectors, sentence_ends, question_vectors, spt, context_words, context_question_answers, answer_vectors, _ = zip(*batch_data)
        ends = list(sentence_ends)
        max_end = max(map(len, ends))
        aends = np.zeros((len(ends), max_end))
        for index, i in enumerate(ends):
            for indexj, x in enumerate(i):
                aends[index, indexj] = x-1
        new_ends = np.zeros(aends.shape+(2,))
        for index, x in np.ndenumerate(aends):
            new_ends[index+(0,)] = index[0]
            new_ends[index+(1,)] = x
        contexts = list(context_vectors)
        max_context_length = max([len(x) for x in contexts])
        context_size = list(np.array(contexts[0]).shape)
        context_size[0] = max_context_length
        final_contexts = np.zeros([len(contexts)]+context_size)
        contexts = [np.array(x) for x in contexts]
        for i, context in enumerate(contexts):
            final_contexts[i, 0:len(context), :] = context
        max_query_length = max(len(x) for x in question_vectors)
        question_size = list(np.array(question_vectors[0]).shape)
        question_size[:1] = [len(question_vectors), max_query_length]
        queries = np.zeros(question_size)
        question_lenghts = np.array(list(zip(range(len(question_vectors)),[len(q)-1 for q in question_vectors])))
        questions = [np.array(q) for q in question_vectors]
        for i, question in enumerate(questions):
            queries[i, 0:len(question), :] = question
        data = {self.context_placeholder: final_contexts, self.input_sentence_endings: new_ends,
                self.question:queries, self.input_question_lengths: question_lenghts, self.gold_standard: answer_vectors}
        return (data, context_words, context_question_answers) if more_data else data

    def init_batch_data(self):
        # 准备验证集
        batch = np.random.randint(self.final_test_data.shape[0], size=self.batch_size * 10)
        batch_data = self.final_test_data[batch]
        self.validation_set, self.val_context_words, self.val_cqas = self.prep_batch(batch_data, True)

    def train(self, iterations, batch_size=128):
        training_iterations = range(0, iterations, batch_size)
        training_iterations = tqdm(training_iterations)
        wordz = []
        for j in training_iterations:
            batch = np.random.randint(self.final_train_data.shape[0], size=batch_size)
            batch_data = self.final_train_data[batch]
            #         print(batch_data)
            self.sess.run([self.optimizer_op], feed_dict=self.prep_batch(batch_data))
            if (j / batch_size) % self.display_step == 0:
                # 计算当前批次准确率
                acc, ccs, tmp_loss, log, con, cor, loc = self.sess.run(
                    [self.corrects_float, self.context_facts, self.total_loss, self.logit, self.context_placeholder,
                     self.corrects_answer, self.logit_bool],
                    feed_dict=self.validation_set)
                # 显示结果
                print("Iter" + str(j / batch_size) + ",Minibach Loss= ", tmp_loss, "Accuracy= ", np.mean(acc))


    def show_pic(self):
        ancr = self.sess.run([self.corrbool, self.logit_bool, self.total_loss, self.logit_score, self.facts_Os, self.weights_1] + self.attend_list+
                        [self.question, self.context_facts, self.question_module_outputs], feed_dict=self.validation_set)
        a = ancr[0]
        n = ancr[1]
        cr = ancr[2]
        attenders = np.array(ancr[6:-3])

        # 上下文中事实的数目
        faq = np.sum(ancr[4], axis=(-1,-2))
        limit = 5
        for question in range(min(limit, self.batch_size)):
            plt.yticks(range(self.passes,0,-1))
            plt.ylabel("Episode")
            plt.xlabel("Question "+str(question+1))
            pltdata = attenders[:,question,:int(faq[question]),0]
            # 显示事实实际存在的信息
            pltdata = (pltdata - pltdata.mean()) / ((pltdata.max() - pltdata.min() + 0.001)) * 256
            plt.pcolor(pltdata, cmap=plt.cm.BuGn, alpha=0.7)
            plt.show()
        # print(list(map((lambda x: x.shape),ancr[3:])),new_ends.shape)

        # 上下文中回答的位置
        indices = np.argmax(n, axis=1)

        # 上下文正确答案的位置
        indicesc = np.argmax(a, axis=1)
        for i,e,cw,cqa in list(zip(indices, indicesc, self.val_context_words, self.val_cqas))[:limit]:
            ccc = " ".join(cw)
            print("TEXT: ", ccc)
            print("QUESTION: "," ".join(cqa[3]))
            print("RESPONSE: ",cw[i],["Correct","Incorrect"][i!=e])
            print("EXPECTED: ",cw[e])
            print()

    def accuracy(self):
        print(np.mean(self.sess.run([self.corrects_float], feed_dict=self.prep_batch(self.final_test_data))[0]))

    def save_model(self):
        print('save start...')
        self.saver.save(self.sess, "model/intelligent_qas.ckpt")
        print('end')

if __name__ == '__main__':
    qas = Intelligent_QAS()
    qas.train(3000)
    qas.save_model()