import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import srio.org.apache.spark.mllib.sampling._
import utils.keel.KeelParser
import java.io.File
import breeze.linalg.split
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.feature.MDLPDiscretizer
import java.nio.channels.MulticastChannel

object BigData {
  def main(args:Array[String]): Unit = {
    // Configuración de Spark
		val conf = new SparkConf()
    conf.set("spark.cores.max","20")
    conf.set("spark.executor.memory","6g")
    conf.set("spark.kryoserializer.buffer.max","512")  
		conf.setAppName("Practica_BigData_Spark")
		
		// Leemos los datasets
		val sc = new SparkContext(conf)
	  val trainData = sc.textFile("hdfs://hadoop-master/user/spark/datasets/ECBDL14_mbd/ecbdl14tra.data")
		val testData = sc.textFile("hdfs://hadoop-master/user/spark/datasets/ECBDL14_mbd/ecbdl14tst.data")
		val parser = new KeelParser(sc, "hdfs://hadoop-master/user/spark/datasets/ECBDL14_mbd/ecbdl14.header")
		
		// Transformamos los datos de RDD[String] RDD[Laberedpoint] para trabajar de forma más facil
		val trainDataLabered = trainData.map{x => parser.parserToLabeledPoint(x)}
	  val testDataLabered = testData.map{x => parser.parserToLabeledPoint(x)}
	  
    // Discretizamos la base de datos para poder luego extraer caracteristicas
    val categoricalFeat: Option[Seq[Int]] = None
    val nBins = 25
    val maxByPart = 10000
    val discretizer = MDLPDiscretizer.train(trainDataLabered, categoricalFeat, nBins, maxByPart) 
    
    val trainDiscrete = trainDataLabered.map(i => LabeledPoint(i.label, discretizer.transform(i.features)))
	  val testDiscrete = testDataLabered.map(i => LabeledPoint(i.label, discretizer.transform(i.features)))
	    
	  // Realizamos la seleccion de caracteristicas
    val featureSelector = new InfoThSelector(new InfoThCriterionFactory("mrmr"),50,6).fit(trainDiscrete)
    val trainSelection = trainDiscrete.map(i => LabeledPoint(i.label, featureSelector.transform(i.features))).cache()
    val testSelection = testDiscrete.map(i => LabeledPoint(i.label, featureSelector.transform(i.features))).cache() 
      
	    
    // Dataset de pruebas
    val dataNoPrep = trainDataLabered
    val dataNoPrepRus = runRUS.apply(trainDataLabered, 1.0, 0.0)
    val dataNoPrepRos50 = runROS.apply(trainDataLabered, 1.0, 0.0, 50)
    val dataNoPrepRos100 = runROS.apply(trainDataLabered, 1.0, 0.0, 100)
    val trainDataDisSel = trainSelection
    val trainDataDisSelRus = runRUS.apply(trainSelection, 1.0, 0.0)
    val trainDataDisSelRos50 = runROS.apply(trainSelection, 1.0, 0.0, 50)
    val trainDataDisSelRos100 = runROS.apply(trainSelection, 1.0, 0.0, 100)
      
    println("DecisionTree - Sin preprocesamiento")
    runDecisionTree(dataNoPrep, testDataLabered, 10, 32)
    println("DecisionTree - Sin preprocesamiento balanceo RUS")
    runDecisionTree(dataNoPrepRus, testDataLabered, 10, 32)
    println("DecisionTree - Sin preprocesamiento balanceo ROS 50%")
    runDecisionTree(dataNoPrepRos50, testDataLabered, 10, 32)
    println("DecisionTree - Sin preprocesamiento balanceo ROS 100%")
    runDecisionTree(dataNoPrepRos100, testDataLabered, 10, 32)
    println("DecisionTree - Discretización y Selec. Caract 50")
    runDecisionTree(trainDataDisSel, testSelection, 10, 32)
    println("DecisionTree - Discretización y Selec. Caract 50 balanceo RUS")
    runDecisionTree(trainDataDisSelRus, testSelection, 10, 32)
    println("DecisionTree - Discretización y Selec. Caract 50 balanceo ROS 50%")
    runDecisionTree(trainDataDisSelRos50, testSelection, 10, 32)
    println("DecisionTree - Discretización y Selec. Caract 50 balanceo ROS 100%")
    runDecisionTree(trainDataDisSelRos100, testSelection, 10, 32)
	    
    println("RandomForest - Sin preprocesamiento")
    runRandomForest(dataNoPrep, testDataLabered, 10, 4, 32)
    println("RandomForest - Sin preprocesamiento balanceo RUS")
    runRandomForest(dataNoPrepRus, testDataLabered, 10, 4, 32)
    println("RandomForest - Sin preprocesamiento balanceo ROS 50%")
    runRandomForest(dataNoPrepRos50, testDataLabered, 10, 4, 32)
    println("RandomForest - Sin preprocesamiento balanceo ROS 100%")
    runRandomForest(dataNoPrepRos100, testDataLabered, 10, 4, 32)
    println("RandomForest - Discretización y Selec. Caract 50")
    runRandomForest(trainDataDisSel, testSelection, 10, 4, 32)
    println("RandomForest - Discretización y Selec. Caract 50 balanceo RUS")
    runRandomForest(trainDataDisSelRus, testSelection, 10, 4, 32)
    println("RandomForest - Discretización y Selec. Caract 50 balanceo ROS 50%")
    runRandomForest(trainDataDisSelRos50, testSelection, 10, 4, 32)
    println("RandomForest - Discretización y Selec. Caract 50 balanceo ROS 100%")
    runRandomForest(trainDataDisSelRos100, testSelection, 10, 4, 32)
      
      
    // Prueba con los paremetros del algoritmo DecisionTree
    val testmDepth = Array(5, 10, 20, 30)
    val testmBins = Array(5, 20, 50, 100)
    for(mD <- testmDepth; mB <- testmBins) runDecisionTree(trainDataDisSelRos100, testSelection, mD, mB)

    // Prueba con los paremetros del algoritmo RandomForest
    val testRFnTree = Array(5, 10, 30, 50)
    val testRFmDepth = Array(4, 10, 20, 30)
    for(nT <- testRFnTree; mD <- testRFmDepth) runRandomForest(trainDataDisSelRos100, testSelection, nT, mD, 32)
  }
  
  def runDecisionTree(train:RDD[LabeledPoint], test:RDD[LabeledPoint], mDepth:Int, mBins:Int){
	  val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "entropy"
    val maxDepth = mDepth
    val maxBins = mBins
    
    val model = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo,
            impurity, maxDepth, maxBins)
            
    val predictions = test.map { point => 
      val prediction = model.predict(point.features) 
      (prediction, point.label) 
    }
          
    val metrics = new MulticlassMetrics(predictions)
    val precision = metrics.precision
    val cm = metrics.confusionMatrix
    println("\nDecisionTree maxDepth = " + mDepth + " maxBins = " + mBins)
    println(cm)
          
    val tpr = cm(0,0)/(cm(0,0) + cm(0,1))
    val tnr = cm(1,1)/(cm(1,0) + cm(1,1))
          
    println("\nTPR = " + tpr + " -- " + "TNR = " + tnr)
    println("Gmean = " + tpr*tnr)
    println("--------------------------------------------------------------")
	}
	
	def runRandomForest(train:RDD[LabeledPoint], test:RDD[LabeledPoint], nTree:Int, mDepth:Int, mBins:Int){
	  val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = nTree // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = mDepth
    val maxBins = mBins

    val model = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
            
    val predictions = test.map { point => 
      val prediction = model.predict(point.features) 
      (prediction, point.label) 
    }
          
    val metrics = new MulticlassMetrics(predictions)
    val precision = metrics.precision
    val cm = metrics.confusionMatrix
    println("\nRamdonForest numTree = " + nTree + " maxDepth = " + mDepth + " maxBins = " + mBins)
    println(cm)
          
    val tpr = cm(0,0)/(cm(0,0) + cm(0,1))
    val tnr = cm(1,1)/(cm(1,0) + cm(1,1))
          
    println("\nTPR = " + tpr + " -- " + "TNR = " + tnr)
    println("Gmean = " + tpr*tnr)
    println("--------------------------------------------------------------")
	}
}