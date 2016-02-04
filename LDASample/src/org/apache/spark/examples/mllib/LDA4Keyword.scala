/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// scalastyle:off println
package org.apache.spark.examples.mllib
import java.text.BreakIterator
import scala.collection.mutable.HashMap
import scala.collection.mutable
import scala.List
import scala.math._
import scopt.OptionParser
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf, Accumulator}
import org.apache.spark.mllib.clustering.{EMLDAOptimizer, OnlineLDAOptimizer, DistributedLDAModel, LDA, LocalLDAModel, LDAModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors, SparseVector, Matrix, DenseMatrix,Matrices}
import org.apache.spark.rdd.RDD
import java.io._
import org.apache.spark.mllib.feature._
import org.apache.spark.sql._
import org.apache.spark.util.Utils
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import scala.swing._
import scala.swing.event._
import org.apache.commons.io.FileUtils
import org.apache.commons.io.filefilter.WildcardFileFilter
import breeze.collection.mutable.{SparseArray}

/**
 * An example Latent Dirichlet Allocation (LDA) app. Run with
 * {{{
 * ./bin/run-example mllib.LDAExample [options] <input>
 * }}}
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object LDA4Keyword {
  val conf = new SparkConf()
    .setAppName(s"LDA4Keyword")
    .setMaster("local[4]")
    .set("spark.driver.maxResultSize", "14g")
  val sc = new SparkContext(conf)
  var ldamodel: LDAModel = null
  var vocab: Map[String, Int] = null
  var vocabArray: Array[String] = null
  var pmiMatrix: Matrix = null
  var topic_num = 100
  var doc_corpus: RDD[(Long, Vector)] = null
  val output = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("result.txt", true)))
  
  
  private case class UI (defaultParams: Params, parser:OptionParser[Params], args:Array[String]) extends MainFrame {
    title = "LDA for Keywords"
    preferredSize = new Dimension(320, 480)
    val query = new TextField { columns = 32 }
    val result = new TextArea { rows = 18; lineWrap = true; wordWrap = true }
    val train = new Button("Train")
    val search = new Button("Search")
    var topicsMat: Matrix = null
    train.enabled = true
    search.enabled = true
    query.text = "東芝 液晶テレビ"
    contents = new BoxPanel(Orientation.Vertical) {
      contents += new Label("query:")
      contents += query
      contents += train
      contents += Swing.HGlue
      contents += search
      contents += result
    }
    
    listenTo(train)
    listenTo(search)
    
    reactions += {
      case ButtonClicked(`train`) => {
        train.enabled = false
        train.text = "Traing..."
        parser.parse(args, defaultParams).map { params =>
          topic_num = params.k
          ldamodel = run(params)
          //println(ldamodel.topicsMatrix.toString())
        }.getOrElse {
          parser.showUsageAsError
          sys.exit(1)
        }
        train.text = "Trained"
        search.enabled = true
      }
      case ButtonClicked(`search`) => {
        result.text = ""
        if(doc_corpus == null){
          doc_corpus = sc.objectFile("corpus")
        }
        if(ldamodel == null){
          ldamodel = LocalLDAModel.load(sc, "myLDAModel")
          vocab = sc.textFile("vocab.txt").map(s => s.split(" ")).map(x => x(0)).zipWithIndex().map(x => (x._1, x._2.toInt)).collect.toMap
          vocabArray = new Array[String](vocab.size)
          vocab.foreach { case (term, i) => vocabArray(i) = term }
        }
        if (ldamodel.isInstanceOf[LocalLDAModel]) {
          val locLDAModel = ldamodel.asInstanceOf[LocalLDAModel]
          topic_num = locLDAModel.k
          if(topicsMat == null){
            val test = locLDAModel.describeTopics().map{ case(idxes,tokens) => idxes.zip(tokens)}
                                                    .map{paires => paires.sortBy(_._1)}
                                                    .map(paires => paires.map(pair => pair._2).toSeq)
                                                    .flatMap(x => x)
                                                    //.zipWithIndex
            
            topicsMat = new DenseMatrix(vocab.size, topic_num, test)
          }
          val query_txt = query.text.trim().split(Array(' ', '　'))
          val doc_list = Seq(query_txt)
          val new_doc = sc.parallelize(doc_list).zipWithIndex().map(_.swap)
                          .map { case (id, tokens) =>
                          // Filter tokens by vocabulary, and create word count vector representation of document.
                          val wc = new HashMap[Int, Int]()
                          tokens.foreach { term =>
                            if (vocab.contains(term)) {
                              val termIndex = vocab(term)
                              wc(termIndex) = wc.getOrElse(termIndex, 0) + 1
                            }
                          }
                          val indices = wc.keys.toArray.sorted
                          val values = indices.map(i => wc(i).toDouble)
                          val sb = Vectors.sparse(vocab.size, indices, values)
                          (id, sb)
                        }
          val new_doc_topics = locLDAModel.topicDistributions(new_doc).map(x => x._2).first()
          val new_topics_words = topicsMat.multiply(new_doc_topics)

          val res = new_topics_words.toArray.zipWithIndex.sortBy(_._1).takeRight(10)
          res.foreach{
            case(score, idx) => 
              //println(idx)
              result.text = vocabArray(idx.toInt) + " : " + score + "\n" + result.text
          }
          val pair_Map = res.map{ case(value, idx) => (idx, value)}.toMap
          val keywords_set = new mutable.HashSet[Int]
          res.map(pair => pair._2).foreach(elem => {
            keywords_set.add(elem)
          })

          val query_Array = new Array[Int](query_txt.length)
          for(i <- 0 to query_txt.length-1){
            keywords_set.add(vocab(query_txt(i)))
            query_Array(i) = vocab(query_txt(i))
          }
          val keywords_Array = keywords_set.toArray
          val value_Array = new Array[Double](keywords_set.size)
          for(i <- 0 to keywords_Array.length-1){
            if(pair_Map.contains(keywords_Array(i)))
              value_Array(i) = pair_Map(keywords_Array(i))
            else
              value_Array(i) = 0d
          }
          
          val input_Array = Array.ofDim[(Array[Int], Array[Double])](1)
          //println(keywords_set.size)
          input_Array(0) = (keywords_Array, value_Array)
          val result_pmi = pmi(doc_corpus,input_Array,vocabArray.length, query_Array,10)
          //result.text = result.text + "\r\n\r\n" + "Average PMI: " + result_pmi(0)._2
          
          output.write(result_pmi(0)._2 + "\n")
          output.write(result.text + "\n")
          output.flush
        }
      }
      case e: event.WindowClosing => {
        sc.stop
        output.close()
      }
    }
  }
  private case class Params(
      input: Seq[String] = Seq.empty,
      cutRate: Double = 0.9,
      k: Int = topic_num,
      maxIterations: Int = 100,
      docConcentration: Double = -1,
      topicConcentration: Double = -1,
      vocabSize: Int = -1,
      stopwordFile: String = "",
      algorithm: String = "online",
      checkpointDir: Option[String] = None,
      checkpointInterval: Int = 10) extends AbstractParams[Params]
  
  val hashingTF = new HashingTF()
  
  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("LDAExample") {
      head("LDAExample: an example LDA app for plain text data.")
      opt[Int]("cutRate")
        .text(s"percentage of keywords for feature. default: ${defaultParams.cutRate}")
        .action((x, c) => c.copy(cutRate = x))
      opt[Int]("k")
        .text(s"number of topics. default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[Int]("maxIterations")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[Double]("docConcentration")
        .text(s"amount of topic smoothing to use (> 1.0) (-1=auto)." +
        s"  default: ${defaultParams.docConcentration}")
        .action((x, c) => c.copy(docConcentration = x))
      opt[Double]("topicConcentration")
        .text(s"amount of term (word) smoothing to use (> 1.0) (-1=auto)." +
        s"  default: ${defaultParams.topicConcentration}")
        .action((x, c) => c.copy(topicConcentration = x))
      opt[Int]("vocabSize")
        .text(s"number of distinct word types to use, chosen by frequency. (-1=all)" +
          s"  default: ${defaultParams.vocabSize}")
        .action((x, c) => c.copy(vocabSize = x))
      opt[String]("stopwordFile")
        .text(s"filepath for a list of stopwords. Note: This must fit on a single machine." +
        s"  default: ${defaultParams.stopwordFile}")
        .action((x, c) => c.copy(stopwordFile = x))
      opt[String]("algorithm")
        .text(s"inference algorithm to use. em and online are supported." +
        s" default: ${defaultParams.algorithm}")
        .action((x, c) => c.copy(algorithm = x))
      opt[String]("checkpointDir")
        .text(s"Directory for checkpointing intermediate results." +
        s"  Checkpointing helps with recovery and eliminates temporary shuffle files on disk." +
        s"  default: ${defaultParams.checkpointDir}")
        .action((x, c) => c.copy(checkpointDir = Some(x)))
      opt[Int]("checkpointInterval")
        .text(s"Iterations between each checkpoint.  Only used if checkpointDir is set." +
        s" default: ${defaultParams.checkpointInterval}")
        .action((x, c) => c.copy(checkpointInterval = x))
      arg[String]("<input>...")
        .text("input paths (directories) to plain text corpora." +
        "  Each text file line should hold 1 document.")
        .unbounded()
        .required()
        .action((x, c) => c.copy(input = c.input :+ x))
    }
    val mainWin = new UI(defaultParams,parser,args)
    mainWin.visible = true
    mainWin.train.doClick()
    mainWin.search.doClick()
    mainWin.close()
    //println(defaultParams.docConcentration)
    //println(defaultParams.topicConcentration)
    /*
    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      parser.showUsageAsError
      sys.exit(1)
    }
    * *
    */
  }
  private def run(params: Params): LDAModel = {
    Logger.getRootLogger.setLevel(Level.WARN)

    // Load documents, and prepare them for LDA.
    //println(params.maxIterations)
    val preprocessStart = System.nanoTime()
    val (corpus, actualNumTokens) =
      preprocess(sc, params.input, params.cutRate, params.vocabSize, params.stopwordFile)
    corpus.cache()
    doc_corpus = corpus
    val test_corpus = sc.objectFile[(Long, org.apache.spark.mllib.linalg.Vector)]("testdata")
    val train_corpus = corpus.subtract(test_corpus)
    val actualCorpusSize = train_corpus.count()
    val actualVocabSize = vocabArray.size
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9
    println()
    println(s"Corpus summary:")
    println(s"\t Training set size: $actualCorpusSize documents")
    println(s"\t Vocabulary size: $actualVocabSize terms")
    println(s"\t Training set size: $actualNumTokens tokens")
    println(s"\t Preprocessing time: $preprocessElapsed sec")
    println()
    
    // Run LDA.
    val lda = new LDA()
    val optimizer = params.algorithm.toLowerCase match {
      case "em" => new EMLDAOptimizer
      // add (1.0 / actualCorpusSize) to MiniBatchFraction be more robust on tiny datasets.
      case "online" => new OnlineLDAOptimizer().setMiniBatchFraction(0.05 + 1.0 / actualCorpusSize)
      											.setOptimizeDocConcentration(true)
      case _ => throw new IllegalArgumentException(
        s"Only em, online are supported but got ${params.algorithm}.")
    }
    lda.setOptimizer(optimizer)
      .setK(params.k)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .setCheckpointInterval(params.checkpointInterval)
    if (params.checkpointDir.nonEmpty) {
      sc.setCheckpointDir(params.checkpointDir.get)
    }
    output.write("maxIteration : " + params.maxIterations + "\n")
    output.write("Training time, log_perplexity, topic PMI, query PMI" + "\n")
    
    val startTime = System.nanoTime()
    //val test_corpus = corpus.sample(withReplacement=false, fraction=0.01)
    //test_corpus.saveAsObjectFile("testdata")
    val ldaModel = lda.run(train_corpus)
    val elapsed = (System.nanoTime() - startTime) / 1e9
    println(s"Finished training LDA model.  Summary:")
    println(s"\t Training time: $elapsed sec")
    
    output.write(elapsed + "\n")
        
    if (ldaModel.isInstanceOf[DistributedLDAModel]) {
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
      val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble
      println(s"\t Training data average log likelihood: $avgLogLikelihood")
      println()
    }
    else{
      val onlineLDAModel = ldaModel.asInstanceOf[LocalLDAModel]
      val logPerplexity = onlineLDAModel.logPerplexity(test_corpus)
      println(s"\t Test data log Perplexity: $logPerplexity")
      println()
      output.write(logPerplexity + "\n")
    }
    // Print the topics, showing the top-weighted terms for each topic.

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    /*
    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map { case (term, weight) => (term, vocabArray(term.toInt), weight) }
    }

    println(s"${params.k} topics:")
    topics.zipWithIndex.foreach { case (topic, i) =>
      println(s"TOPIC $i")
      topic.foreach { case (term, word, weight) =>
        println(s"$term\t$word\t$weight")
      }
      println()
    }
    * 
    */

    //Calculate Average PMI
    val total_pmi = sc.accumulator(0d)
    
    val topic_pmi = pmi(corpus, topicIndices, actualVocabSize)
    topic_pmi.foreach(pair => {
      //println(s"PMI of Topic " + pair._1 + " " + pair._2)
      total_pmi += pair._2
    } 
    )
    
    val average_pmi = total_pmi.value / topic_pmi.length
    
    println("Topic Number: " + topicIndices.length)
    println("Average PMI is: " + average_pmi)
    
    output.write(average_pmi + "\n")
    output.flush()
    
    deleteDirectory("myLDAModel")
    //sc.stop()
    ldaModel.save(sc, "myLDAModel")
    //println(ldaModel.docConcentration)
    //println(ldaModel.topicConcentration)
    
    System.gc()
    return ldaModel
  }

  def seqsum(start:Int=1, limit:Int, dif:Int=1): Long ={ 
    val n = ((limit-start) / dif) + 1
    (start+limit)*n/2
  }
  
  def pmi(docs:RDD[(Long, Vector)],topicIndices: Array[(Array[Int], Array[Double])], vocSize: Int, query: Array[Int]=null, retrieved: Int=0): Array[(Int,Double)] ={
    //Load PMI Matrix if exists
    //var pMatrix = Matrices.speye(vocSize)
    val tval= seqsum(limit=vocSize)
    //val pArray = new SparseArray[Accumulator[Double]](size=seqsum(limit=vocSize))
    val topics = topicIndices.map { case (term_idxes, termWeights) => term_idxes }
    var kw_map: Array[Map[Int, Double]] = null
    //var val_total: Double = 0d
    //for query PMI calculation
  
    if(query != null){
      kw_map = topicIndices.map { case (term_idxes, termWeights) => term_idxes.zip(termWeights).toMap }
      //val_total = kw_map(0).values.sum
    }

    
    val pMap = new HashMap[Long, Accumulator[Double]]
    val topicWords = topics.flatMap{arr => arr}.toSet.toArray.sorted//indexes of tokens
    
    //initialize pMap
    for(i <-0 to topicWords.length-1){
      val ssum: Long = seqsum(limit=topicWords(i))
      for(j <-0 to i){
        val idx_j = topicWords(j).toLong
        //println("key: " + (ssum+idx_j) + " = " + topicWords(i) + " "+ vocabArray(topicWords(i)) +" , " + idx_j + " " +vocabArray(topicWords(j)))
        pMap += ((ssum+idx_j) -> sc.accumulator[Double](0d))
        //pMap(ssum+idx_j) += 1
      }
    }
    //update pMap
    docs.foreach { case (docid, tokens) =>
      val tokenids = tokens.toSparse.indices.sorted
      for (i <- 0 to tokenids.length-1){
        //token's Index
        val idx_i = tokenids(i).toInt
        if(topicWords.contains(idx_i)){
          val ssum = seqsum(limit=idx_i)
  	      for(j <- 0 to i){
  	        var idx_j = tokenids(j).toInt
  	        if(topicWords.contains(idx_j)){
  	          //println(ssum + ", " + idx_j)
  	          val key = ssum + idx_j
  	          pMap(key) += 1
  	        }
	        }
        }
      }
    }
    //Standardization
    //pMap.foreach(elem => elem._2.setValue(elem._2.value / docsCount))
    
    val pmi = topics.zipWithIndex.map{ case(tokens,idx) => 
      val tokens_sorted = tokens.sorted
      var pmi_total = 0d
      var weight = 1d
      var pair_num = 0L
      if(query != null)
        pair_num = 1//(2*tokens.length - query.length - retrieved) * query.length
      else
        pair_num = seqsum(limit=tokens.length)-tokens.length
        
      val logdocsCount = log(docs.count)
      //println("topic id : " + idx)
      for(i <- 0 to tokens.length-1){
        val ssum = seqsum(limit=tokens_sorted(i))
        //println(tokens(i))
        for(j <-0 to i){
          if(i != j){
            if(query != null){
              if(query.contains(tokens_sorted(i)) && !query.contains(tokens_sorted(j))){
                weight = kw_map(0)(tokens_sorted(j))// / val_total
                //println(vocabArray(tokens_sorted(j)) + " : " + weight)
              }
              else if(!query.contains(tokens_sorted(i)) && query.contains(tokens_sorted(j))){
                weight = kw_map(0)(tokens_sorted(i))// / val_total
                //println(vocabArray(tokens_sorted(i)) + " : " + weight)
              }
              else
                weight = 0d
            }
            if(weight != 0 && pMap(ssum+tokens_sorted(j)).value > 0)
          	  pmi_total = (pmi_total + weight * 
          	  (log(pMap(ssum+tokens_sorted(j)).value) + logdocsCount - 
          	      (log(pMap(ssum+tokens_sorted(i)).value) + 
          	          log(pMap(seqsum(limit=tokens_sorted(j))+tokens_sorted(j)).value))))
          }
        }
      }
      (idx, pmi_total / pair_num)
    }
    pmi
    //Array.ofDim[(Int,Double)](1,1)
  }

 
  /**
   * calculate the hashcode of a string object
   * @author charles
   * 
   */
  def hashing(x:String): Int = {
    hashingTF.indexOf(x)
  }
  
  /**
   * delete the folder
   */
  def deleteDirectory(dn:String) = {
    try{
      val file = new File(dn)
      if(file.exists())
    	  FileUtils.deleteDirectory(file)
    }
    catch{
      case e:IOException => println(e.toString())
    }
  }
  
  /**
   * Load documents, tokenize them, create vocabulary, and prepare documents as term count vectors.
   * @return (corpus, vocabulary as array, total token count in corpus)
   */
  private def preprocess(
      sc: SparkContext,
      paths: Seq[String],
      cutRate: Double,
      vocabSize: Int,
      stopwordFile: String): (RDD[(Long, Vector)], Long) = {
    // Get dataset of document texts
    // One document per line in each text file. If the input consists of many small files,
    // this can result in a large number of small partitions, which can degrade performance.
    // In this case, consider using coalesce() to create fewer, larger partitions.
    val textRDD: RDD[String] = sc.textFile(paths.mkString(","))
    
    val docs: RDD[Seq[String]] = textRDD.map(_.split(" ").toSeq)
    val hash2Word: Map[Int, String] = docs.flatMap{doc => doc}
                                             .map{word => hashing(word) -> word}
                                             .distinct()
                                             .collect()
                                             .toMap
    //val tt = docs.flatMap{doc => doc}
    val tf:RDD[Vector] = hashingTF.transform(docs)
    tf.cache()
    val idf = new IDF(minDocFreq = 3).fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)
    //idx is the hashcode of the term
    val TF_IDF_IDX = tfidf.map(vec => vec.toSparse.indices)
                          .flatMap(idx => idx)
    val TF_IDF_VAL = tfidf.map(vec => vec.toSparse.values)
                                          .flatMap(value => value)
    val mean_TFIDF = TF_IDF_IDX.zip(TF_IDF_VAL)
                                      .mapValues(x => (x, 1))
                                      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
                                      .map(pair => pair._1 -> (pair._2._1 / pair._2._2))
                                      
    val TF_IDF_SUM:Double = mean_TFIDF.map(pair => pair._2).sum
    
    val percent_TFIDF_Sorted = mean_TFIDF.map(pair => pair._1 -> (pair._2 / TF_IDF_SUM))
                                      .sortBy(_._2,false)
    var total:Double = 0d
    var count:Int = 0
    val checkNG = (x:(Int,Double)) => {
      try{
          total = total + x._2
          if(total <= cutRate)
            count = count +1
      }
      catch{
        case ex: Exception => {println(ex);false}
      }
    }
    percent_TFIDF_Sorted.collect().foreach(checkNG)
    val selected_feature = percent_TFIDF_Sorted.take(count)
    val selected_feature_map = selected_feature.toMap
    val selected_feature_idx = selected_feature.map(pair => pair._1)
                                                .toSet
 

    val summary: MultivariateStatisticalSummary = Statistics.colStats(tfidf)
    
    // Split text into words
    val tokenizer = new SimpleTokenizer(sc, stopwordFile)
    val tokenized: RDD[(Long, IndexedSeq[String])] = textRDD.zipWithIndex().map { case (text, id) =>
      id -> tokenizer.getWords(text)
    }
    tokenized.cache()
    // Counts words: RDD[(word, wordCount)]
    val wordCounts: RDD[(String, Long)] = tokenized
      .flatMap { case (_, tokens) => tokens.map(_ -> 1L) }
      .filter(pair => if(selected_feature_idx.contains(hashing(pair._1))) true else false)//check whether the token is selected by mean TF-IDF
      .reduceByKey(_ + _)
    wordCounts.cache()
    //println(wordCounts.count())
    val fullVocabSize = wordCounts.count()
    // Select vocab
    //  (vocab: Map[word -> id], total tokens after selecting vocab)
    val (selectedTokenCount: Long) = {
      val tmpSortedWC: Array[(String, Long)] = if (vocabSize == -1 || fullVocabSize <= vocabSize) {
        // Use all terms
        wordCounts.collect().sortBy(-_._2)
      } else {
        // Sort terms to select vocab
        wordCounts.sortBy(_._2, ascending = false).take(vocabSize)
      }
      //generate the vocabulary set with indexing
      vocab = tmpSortedWC.map(_._1).zipWithIndex.toMap
      (tmpSortedWC.map(_._2).sum)
    }
    
    val documents = tokenized.map { case (id, tokens) =>
      // Filter tokens by vocabulary, and create word count vector representation of document.
      val wc = new HashMap[Int, Int]()
      tokens.foreach { term =>
        if (vocab.contains(term)) {
          val termIndex = vocab(term)
          wc(termIndex) = wc.getOrElse(termIndex, 0) + 1
        }
      }
      val indices = wc.keys.toArray.sorted
      val values = indices.map(i => wc(i).toDouble)
      val sb = Vectors.sparse(vocab.size, indices, values)
      (id, sb)
    }
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("vocab.txt")))
    vocabArray = new Array[String](vocab.size)
    vocab.foreach { case (term, i) => 
      vocabArray(i) = term
      }
    vocabArray.foreach { x => writer.write(x + "\n") }
    writer.close()
    deleteDirectory("corpus")
    documents.saveAsObjectFile("corpus")
    (documents, selectedTokenCount)
  }
}
/**
 * Simple Tokenizer.
 *
 * TODO: Formalize the interface, and make this a public class in mllib.feature
 */
private class SimpleTokenizer(sc: SparkContext, stopwordFile: String) extends Serializable {
  private val stopwords: Set[String] = if (stopwordFile.isEmpty) {
    Set.empty[String]
  } else {
    val stopwordText = sc.textFile(stopwordFile).collect()
    stopwordText.flatMap(_.stripMargin.split("\\s+")).toSet
  }
  // Matches sequences of Unicode letters
  private val allWordRegex = "^(\\p{L}*)$".r
  // Ignore words shorter than this length.
  private val minWordLength = 3
  def getWords(text: String): IndexedSeq[String] = {
    val words = new mutable.ArrayBuffer[String]()
    for(w <- text.split(" ")) {
      words+=w
    }
    words
  }
  /*
  def getWords(text: String): IndexedSeq[String] = {
    val words = new mutable.ArrayBuffer[String]()
    // Use Java BreakIterator to tokenize text into words.
    val wb = BreakIterator.getWordInstance
    wb.setText(text)
    // current,end index start,end of each word
    var current = wb.first()
    var end = wb.next()
    while (end != BreakIterator.DONE) {
      // Convert to lowercase
      val word: String = text.substring(current, end).toLowerCase
      // Remove short words and strings that aren't only letters
      word match {
        case allWordRegex(w) if w.length >= minWordLength && !stopwords.contains(w) =>
          words += w
        case _ =>
      }
      current = end
      try {
        end = wb.next()
      } catch {
        case e: Exception =>
          // Ignore remaining text in line.
          // This is a known bug in BreakIterator (for some Java versions),
          // which fails when it sees certain characters.
          end = BreakIterator.DONE
      }
    }
    words
  }
  */
  
}
// scalastyle:on println