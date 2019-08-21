##### Installation

 1. 可以查阅[marian官方博客](https://marian-nmt.github.io/docs/)中的Installation部分。
 2. 在新服务器上安装marian环境也可以参考文档：[使用开源的神经机器翻译框架Marian搭建机器翻译引擎](https://gitlab.tmxmall.com/tmxmall_nmt/marian_nmt/blob/master/%E4%BD%BF%E7%94%A8%E5%BC%80%E6%BA%90%E7%9A%84%E7%A5%9E%E7%BB%8F%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91%E6%A1%86%E6%9E%B6Marian%E6%90%AD%E5%BB%BA%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91%E5%BC%95%E6%93%8E.md)，该文档中涉及到的所需下载的内容在当前公司的两台服务器中对应的目录分别为：
 - `192.168.0.151`：`/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wangxiuwan/`
 - `192.168.0.174`：`/media/wangxiuwan/`
 3. 另外在工作随笔的文档中也有关于marian安装的详细记录：
 - `2019.2.25-marian环境安装.txt`
 - `2019.4.8~2019.4.12-174marian安装-HanLPdemo.txt`
 - 2019-8-12已经安装了github上最新的marian-dev，建议后续训练全部在这个新版本上进行。
##### Training
 1. 训练前语料预处理过程
 			- 预处理主要包含6个过程：`tokenize、clean、train truecaser、apply truecaser、train BPE、apply BPE.`
 			- 详细代码以及注释如下（6个备注）：
```shell
#!/bin/bash -v

ROOT=/media/wangxiuwan/marian
TOOL=$ROOT/examples
# 备注1：一般来说，没有特殊改动，只需要变更TMXMALL这个语料的路径，即可进行语料预处理。
TMXMALL=/media/tmxmall/marian_nmt/general.gen.0723
MIDDLE=$TMXMALL/middle

mkdir -p $MIDDLE
mkdir -p $MIDDLE/model

# suffix of source language files
SRC=zh

# suffix of target language files
TRG=en

# number of merge operations  
# 备注2：该参数与词汇表大小的参数设置关联，根据wmt论文，该参数略小于词汇表大小参数即设置合理，目前词汇表大小为36000，因此bpe_operations设置为32000.
bpe_operations=32000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$TOOL/tools/moses-scripts

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=$TOOL/tools/subword-nmt/subword_nmt

# tokenize  备注3：中文语料已经用hanlp做过分词处理，因此此处只针对英文语料。
for prefix in train valid test
do
    cp $TMXMALL/$prefix.$SRC $MIDDLE/$prefix.tok.$SRC

    test -f $TMXMALL/$prefix.$TRG || continue

    cat $TMXMALL/$prefix.$TRG \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > $MIDDLE/$prefix.tok.$TRG
done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
# 备注4：这里的过滤比例1 100可以根据需求自行调整。
mv $MIDDLE/train.tok.$SRC $MIDDLE/train.tok.uncleaned.$SRC
mv $MIDDLE/train.tok.$TRG $MIDDLE/train.tok.uncleaned.$TRG
$mosesdecoder/scripts/training/clean-corpus-n.perl $MIDDLE/train.tok.uncleaned $SRC $TRG $MIDDLE/train.tok 1 100

# train truecaser   备注5：因为中文没有大小写区分，因此此处也只针对英文做truecaser处理
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $MIDDLE/train.tok.$TRG -model $MIDDLE/model/tc.$TRG

# apply truecaser (cleaned training corpus)
for prefix in train valid test
do
    cp $MIDDLE/$prefix.tok.$SRC $MIDDLE/$prefix.tc.$SRC
    test -f $MIDDLE/$prefix.tok.$TRG || continue
    $mosesdecoder/scripts/recaser/truecase.perl -model $MIDDLE/model/tc.$TRG < $MIDDLE/$prefix.tok.$TRG > $MIDDLE/$prefix.tc.$TRG
done

# train BPE    备注6：因为中文和英文不属于同一个语系，根据wmt的相关论文，BPE处理分开效果更好。如果是英德，属于同一个语系，可进行share的BPE处理（相关脚本可参考marian的官方 example）。
cat $MIDDLE/train.tc.$SRC | $subword_nmt/learn_bpe.py -s $bpe_operations > $MIDDLE/model/$SRC.bpe
cat $MIDDLE/train.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > $MIDDLE/model/$TRG.bpe

# apply BPE
for prefix in train valid test
do
    $subword_nmt/apply_bpe.py -c $MIDDLE/model/$SRC.bpe < $MIDDLE/$prefix.tc.$SRC > $MIDDLE/$prefix.bpe.$SRC
    test -f $MIDDLE/$prefix.tc.$TRG || continue
    $subword_nmt/apply_bpe.py -c $MIDDLE/model/$TRG.bpe < $MIDDLE/$prefix.tc.$TRG > $MIDDLE/$prefix.bpe.$TRG
done
```
 2. 生成词汇表（2个备注）
```shell
#!/bin/bash -v

ROOT=/media/wangxiuwan/marian
MARIAN=$ROOT/build
TRANSFORMER=$ROOT/examples/transformer/9000_dataset_dev
TMXMALL=/media/tmxmall/marian_nmt/general.gen.0723/middle
VOCAB=$TRANSFORMER/model_vocab_big
# if we are in WSL, we need to add '.exe' to the tool names
if [ -e "/bin/wslpath" ]
then
    EXT=.exe
fi

MARIAN_VOCAB=$MARIAN/marian-vocab$EXT

# set chosen gpus
GPUS=0
if [ $# -ne 0 ]
then
    GPUS=$@
fi
echo Using GPUs: $GPUS

#mkdir -p $VOCAB  备注1：有时候从训练脚本中启动生成词汇脚本，会在训练脚本中创建词汇表所需目录。

# create vocabulary 备注2：若是share BPE，可以生成share vocabulary，具体脚本可查阅marian官方demo。
if [ ! -e "$VOCAB/vocab.zh.yml" ]
then
    $MARIAN_VOCAB --max-size 36000 <$TMXMALL/train.bpe.zh>  $VOCAB/vocab.zh.yml
    $MARIAN_VOCAB --max-size 36000 <$TMXMALL/train.bpe.en>  $VOCAB/vocab.en.yml
fi

```

 3. 训练脚本解析	
```shell
#!/bin/bash -v
ROOT=/media/wangxiuwan/marian
MARIAN=$ROOT/build
TRANSFORMER=$ROOT/examples/transformer/9000_dataset_dev
TMXMALL=/media/tmxmall/marian_nmt/general.gen.0723/middle   #经过BPE处理的语料

# if we are in WSL, we need to add '.exe' to the tool names
if [ -e "/bin/wslpath" ]
then
    EXT=.exe
fi

MARIAN_TRAIN=$MARIAN/marian$EXT
MARIAN_DECODER=$MARIAN/marian-decoder$EXT
MARIAN_VOCAB=$MARIAN/marian-vocab$EXT
MARIAN_SCORER=$MARIAN/marian-scorer$EXT

# set chosen gpus
GPUS=0
if [ $# -ne 0 ]
then
    GPUS=$@
fi
echo Using GPUs: $GPUS

if [ ! -e $MARIAN_TRAIN ]
then
    echo "marian is not installed in $MARIAN, you need to compile the toolkit first"
    exit 1
fi

if [ ! -e $ROOT/examples/tools/moses-scripts ] || [ ! -e $ROOT/examples/tools/subword-nmt ] || [ ! -e $ROOT/examples/tools/sacreBLEU ]
then
    echo "missing tools in ../tools, you need to download them first"
    exit 1
fi

if [ ! -e $TMXMALL/train.bpe.zh ]
then
    sh /$TRANSFORMER/scripts/preprocess-data.sh
fi

mkdir -p $TRANSFORMER/model_vocab_big
# create common vocabulary
if [ ! -e $TRANSFORMER/model_vocab_big/vocab.zh.yml ]
then
    sh /$TRANSFORMER/tmxmall-create-vocab.sh
fi

# rm model  备注1：首先清除之前的训练历史模型
if [ -d "$TRANSFORMER/model_zhen" ]
then
    rm -r $TRANSFORMER/model_zhen
fi

mkdir -p $TRANSFORMER/model_zhen    #模型存放目录

mkdir -p $TRANSFORMER/tmxmall_valid_data  #验证输出文件存放目录

# train model
if [ ! -e "$TRANSFORMER/model_zhen/model.npz" ]
then
    $MARIAN_TRAIN \
        --model $TRANSFORMER/model_zhen/model.npz --type transformer \
        --train-sets $TMXMALL/train.bpe.zh $TMXMALL/train.bpe.en \
        --max-length 100 \
        --vocabs $TRANSFORMER/model_vocab_big/vocab.zh.yml $TRANSFORMER/model_vocab_big/vocab.en.yml \
        --mini-batch-fit -w 6000 --maxi-batch 1000 \
        --early-stopping 40 --cost-type=ce-mean-words \
        --valid-freq 5000 --save-freq 5000 --disp-freq 1000 \
        --valid-metrics ce-mean-words perplexity translation \
        --valid-sets $TMXMALL/valid.bpe.zh $TMXMALL/valid.bpe.en \
        --valid-script-path "bash $TRANSFORMER/scripts/validate_zhen.sh" \
        --valid-translation-output $TRANSFORMER/tmxmall_valid_data/valid.en.output --quiet-translation \
        --valid-mini-batch 16 \
        --beam-size 6 --normalize 0.6 \
        --overwrite --keep-best \
        --log $TRANSFORMER/model_zhen/train.log --valid-log $TRANSFORMER/model_zhen/valid.log \
        --enc-depth 6 --dec-depth 6 \
        --transformer-heads 8 \
        --transformer-postprocess-emb d \
        --transformer-postprocess dan \
        --transformer-dropout 0.1 --label-smoothing 0.1 \
        --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
        --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
        --tied-embeddings \
        --devices $GPUS --sync-sgd --seed 1111 \
        --exponential-smoothing
fi
```
1）普通训练
普通训练参考上述脚本即可，主要注意以下参数的设置：

 - 训练集：train-sets
 - 词汇表：vocabs
 - 验证集：valid-sets
 - 验证脚本：valid-script-path
 - 验证集输出：valid-translation-output
 - 训练日志存放：log
 - 验证日志存放：valid-log
 
2）增量训练（不更换语料）
 
 - 使用情况：训练到Up. 45000时，服务器断电，期望在这个基础上继续训练。
 - 修改1：if [ -e "预训练模型路径" ]
 - 修改2：--pretrained-model 需加载的预训练模型路径

3）增量训练（更换语料）

- 使用情况：训练了一个通用领域的模型，以此为基础在垂直领域的语料上训练新的模型。
- 在上述的两处修改的基础上，将--model的存放路径设置为新模型存放路径。
- 关于加载预训练模型，官方解释如下：
Marian has two ways of reusing models and weights:
just via --model path/to/model.npz. If you copy your model to a new folder and set the option to point to that model. It's going to reload the model from the path. It's also going to overwrite it during the next checkpoint. This overrides the model parameters with the model parameters from the file, so you cannot change architectures between continued trainings. This method also works well for normal continued training. So you can interrupt your running training, change the training corpus and run the same command you used before for the training to resume. In the case where you change the training files you would want it to not restore the corpus positions which can be set with --no-restore-corpus. You can also change other training parameters like learning-rate or early-stopping criteria.
via --pretrained-model path/to/model.npz this will load weight matrices from model.npz that match in name corresponding parameters from your architecture. This is more flexible than the method above as it allows you to mix model types. For instance you can initialize the decoder of a RNN encoder-decoder translation model with a RNN language model or deep models with shallow models. This can be used for domain-adaptation or transfer-learning. Non-matching parameters will be initialized randomly. This is a method you should only choose with different model types when the first one is not working for you and you have a reason to go for partial initialization. It is quite safe with matching model parameters.

4. 训练常见问题

 - 关于训练参数的详细解释可查阅[官方博客](https://marian-nmt.github.io/docs/)的Command-line options部分内容。
 - 服务器内存不够充分可添加：--keep-best，只保留最新的模型
 - shuffling sentences to temp files: -T,--tempdir TEXT=/tmp        该参数默认为/tmp，也可以指定其他目录，注意查看是否内存充足，不足时会抛出error writing to file。这个目录下的文件读到内存后会被快速删除，所以目录下看不到内容。[tmp目录说明](https://github.com/marian-nmt/marian-dev/issues/480)
 - 训练是怀疑bleu值，可查看验证目录下的输出：相关参数为--valid-translation-output 
 - 英中和中英模型训练时validate脚本不同，注意设置。
 - 绘制训练曲线：/media/wangxiuwan/marian/examples/tools/trainLogCurve.py
 - 训练语料过大（1亿8千万句对时，可能会报error reading from file），该问题可能为marian缓存机制bug，目前尚未修复，可设置--shuffle-in-ram正常训练。

##### Translation
translation有两种方式，marian-decoder和marian-server

```shell
#marian-decoder
cat $TMXMALL/$prefix \
        | $MARIAN_DECODER -c $TRANSFORMER/model_zhen/model.npz.best-translation.npz.decoder.yml -m $TRANSFORMER/model_zhen/model.npz.best-translation.npz -d $GPUS -b 6 -n  -w 6000 \
        | sed 's/\@\@ //g' \
        | $ROOT/examples/tools/moses-scripts/scripts/recaser/detruecase.perl \
        | $ROOT/examples/tools/moses-scripts/scripts/tokenizer/detokenizer.perl -l en \
         > $TRANSFORMER/tmxmall_valid_data/$prefix.en.output
 #marian-server
 nohup \
../../../build/marian-server --port 8089 --devices 1 \
-c model_enzh/model.npz.best-translation.npz.decoder.yml \
-b 6 \
-m model_enzh/model.test.npz \
-v model_vocab_big/vocab.en.yml model_vocab_big/vocab.zh.yml \
&>0712_enzh_web_nohup.out&	
```
相关参数说明可以查看marian官方博客。
##### Scorer
marian-scorer在官方的wmt2017-transformer的demo中有使用，在151服务器上跑过中英模型，相当于8个模型resemble，bleu值比单模型高2个点左右。
```
/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wangxiuwan/marian/examples/wmt2017-transformer/zhen-run-me.sh
```
