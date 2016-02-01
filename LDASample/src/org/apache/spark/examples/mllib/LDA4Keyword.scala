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
    .set("spark.driver.maxResultSize", "10g")
  val sc = new SparkContext(conf)
  var ldamodel: LDAModel = null
  var vocab: Map[String, Int] = null
  var vocabArray: Array[String] = null
  var pmiMatrix: Matrix = null
  val topic_num = 3
  
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
        if(ldamodel == null){
          ldamodel = LocalLDAModel.load(sc, "myLDAModel")
          vocab = sc.textFile("vocab.txt").map(s => s.split(" ")).map(x => x(0)).zipWithIndex().map(x => (x._1, x._2.toInt)).collect.toMap
          vocabArray = new Array[String](vocab.size)
          vocab.foreach { case (term, i) => vocabArray(i) = term }
        }
        if (ldamodel.isInstanceOf[LocalLDAModel]) {
          val locLDAModel = ldamodel.asInstanceOf[LocalLDAModel]
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
                          val wc = new mutable.HashMap[Int, Int]()
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
              result.text = vocabArray(idx.toInt) + " : " + score + "\r\n" + result.text
          } 
        }
      }
      case e: event.WindowClosed => {
        sc.stop
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
    val startTime = System.nanoTime()
    //val test_corpus = corpus.sample(withReplacement=false, fraction=0.01)
    //test_corpus.saveAsObjectFile("testdata")
    val ldaModel = lda.run(train_corpus)
    val elapsed = (System.nanoTime() - startTime) / 1e9
    println(s"Finished training LDA model.  Summary:")
    println(s"\t Training time: $elapsed sec")
    if (ldaModel.isInstanceOf[DistributedLDAModel]) {
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
      val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble
      println(s"\t Training data average log likelihood: $avgLogLikelihood")
      println()
    }
    /*
    else{
      val onlineLDAModel = ldaModel.asInstanceOf[LocalLDAModel]
      val logPerplexity = onlineLDAModel.logPerplexity(test_corpus)
      println(s"\t Test data log Perplexity: $logPerplexity")
      println()
    }
    * 
    */
    // Print the topics, showing the top-weighted terms for each topic.

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
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

    //Calculate PMI
    val topic_pmi = pmi(corpus, topicIndices, actualVocabSize)
    topic_pmi.foreach(pair => {
      println(s"PMI of Topic $pair._1 : $pair._2")
    } 
    )
    
    deleteDirectory("myLDAModel")
    //sc.stop()
    ldaModel.save(sc, "myLDAModel")
    println(ldaModel.docConcentration)
    println(ldaModel.topicConcentration)
    
    System.gc()
    return ldaModel
  }

  def seqsum(start:Int=1, limit:Int, dif:Int=1): Int ={
    if(abs(limit-start) < abs(dif) || (start+dif) > limit) return(0)    
    val n = ((limit-start) / dif) + 1
    (start+limit)*n/2
  }
  
  def pmi(docs:RDD[(Long, Vector)],topicArry: Array[(Array[Int], Array[Double])], vocSize: Int): Array[(Int,Double)] ={
    //Load PMI Matrix if exists
    //var pMatrix = Matrices.speye(vocSize)
    val pArray = new SparseArray[Accumulator[Double]](seqsum(limit=vocSize))
    val topics = topicArry.map { case (term_idxes, termWeights) => term_idxes }
    val topicWords = topics.flatMap{arr => arr}.toSet.toArray.sorted//indexes of tokens
    val wordsCount = seqsum(limit=topicWords.length)
    val indx = new Array[Int](wordsCount)
    val elem = new Array[Accumulator[Double]](wordsCount)
    
    //initialize pArray
    for(i <-0 to topicWords.length-1){
      val ssum = seqsum(limit=topicWords(i))
      for(j <-0 to i){
        val idx_j = topicWords(j)
        val idx = seqsum(limit=i)+j
        //println(ssum)
        indx(idx) = ssum+idx_j
        elem(idx) = sc.accumulator[Double](0)
        //if(!pArray.contains(ssum+idx_j))
        //pArray.update(ssum+idx_j, sc.accumulator[Double](0))
      }
    }
    pArray.use(indx, elem, elem.length)
    
    //val tes =new SparseArray()
    //update pArray
    docs.foreach { case (docid, tokens) =>
      val tokenids = tokens.toSparse.values.sorted
      for (i <- 0 to tokenids.length-1){
        //token's Index
        val idx_i = tokenids(i).toInt
        if(topicWords.contains(idx_i)){
          val ssum = seqsum(limit=idx_i)
	      for(j <- 0 to i){
	        var idx_j = tokens(j).toInt
	        if(topicWords.contains(idx_j)){
	          println(ssum + ", " + idx_j)
	          pArray(ssum+idx_j) += 1
	        }
	      }
        }
      }
    }
    //Standardization
    pArray.foreach(elem => if(elem != null) elem.setValue(elem.value / docs.count))
    
    val pmi = topics.zipWithIndex.map{ case(tokens,idx) => 
      val tokens_sorted = tokens.sorted
      var pmi_total = 0d
      println("topic id : " + idx)
      for(i <- 0 to tokens.length){
        val ssum = seqsum(limit=tokens_sorted(i))
        println(tokens(i))
        for(j <-0 to i){
          if(i != j)
        	  pmi_total = pmi_total + 
        	  log(pArray(ssum+tokens_sorted(j)).value / 
        	      (pArray(ssum+tokens_sorted(i)).value * pArray(seqsum(limit=tokens_sorted(j))+tokens_sorted(j)).value))
        }
      }
      (idx, pmi_total / tokens.length)
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
      .filter(pair => if(selected_feature_idx.contains(hashing(pair._1))) true else false)//check whether the token is selected
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
      val wc = new mutable.HashMap[Int, Int]()
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