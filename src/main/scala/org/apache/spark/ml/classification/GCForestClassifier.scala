package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasSeed, HasTreeNumCol}
import org.apache.spark.ml.tree.GCForestParams
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, MLWritable}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by chengli on 3/3/17.
  */
class GCForestClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, GCForestClassifier, GCForestClassificationModel]
  with DefaultParamsWritable with GCForestParams {

  def this() = this(Identifiable.randomUID("gcf"))

  def extractMatrixRDD(dataset: Dataset[_], windowWidth: Int, windowHeight: Int): Dataset[_] = {
    require(getDataSize.length == 2, "You must set Image resolution by setDataSize")
    val width = getDataSize(0)
    val height = getDataSize(1)
    require(width >= windowWidth, "The width of image must be greater than window's")
    require(height >= windowHeight, "The heigth of image must be greater than window's")
    val vector2Matrix = udf { (features: Vector, w: Int, h: Int) =>
      new DenseMatrix(w, h, features.toArray)
    }

    val windowInstances = dataset.withColumn($(featuresCol), vector2Matrix(col($(featuresCol))))
      .rdd.zipWithIndex.flatMap {
        case (row, index) =>
          val rows = ArrayBuffer[Row]()
          val featureIdx = row.fieldIndex($(featuresCol))
          val matrix = row.getAs[DenseMatrix]($(featuresCol))
          val newSchema = row.schema
            .add(StructField($(windowId), LongType))
            .add(StructField($(instanceId), LongType))

          // TODO should be optimized
          Range(0, width - windowWidth + 1).foreach { x_offset =>
            Range(0, height - windowHeight + 1).foreach { y_offset =>
              val features = ArrayBuffer[Double]()
              Range(0, windowWidth).foreach { x =>
                Range(0, windowHeight).foreach { y =>
                  features += matrix(x + x_offset, y + y_offset)
                  val newRow = ArrayBuffer[Any](row.toSeq)
                  newRow(featureIdx) = new DenseVector(features.toArray)
                  newRow += (x_offset * (width - windowWidth + 1) + y_offset * (height - windowHeight + 1)).toLong
                  newRow += index
                  rows += new GenericRowWithSchema(newRow.toArray, newSchema)
                }
              }
            }
          }
          rows
    }

    dataset.sparkSession.createDataset(windowInstances)
  }

  def featureTransform(dataset: Dataset[_], rfc: RandomForestClassifier)
  : (Dataset[_], RandomForestClassificationModel) = {
    val schema = dataset.schema
    val sparkSession = dataset.sparkSession
    var out = sparkSession.emptyDataset[Row]
    val splits = MLUtils.kFold(dataset.toDF.rdd, $(numFolds), $(seed))
    splits.zipWithIndex.foreach { case ((training, validation), splitIndex) =>
      val trainingDataset = sparkSession.createDataFrame(training, schema).cache
      val validationDataset = sparkSession.createDataFrame(validation, schema).cache
      val model = rfc.fit(trainingDataset)
      trainingDataset.unpersist()
      out = out.union(
        model.transform(validationDataset)
          .drop($(featuresCol))
          .withColumnRenamed($(probabilityCol), $(featuresCol))
      )
      validationDataset.unpersist()
    }
    (out, rfc.fit(dataset))
  }

  def genRFClassifier(rfType: String,
                          treeNum: Int,
                          minInstancePerNode: Int
                         ): RandomForestClassifier = {
    val rf = rfType match {
      case "rf" => new RandomForestClassifier()
      case "crtf" => new CompleteRandomTreeForestClassifier()
    }

    rf.setNumTrees(treeNum)
      .setMaxBins(Int.MaxValue)
      .setMaxDepth(30)
      .setMinInstancesPerNode(minInstancePerNode)
      .setFeatureSubsetStrategy("sqrt")
  }

  def concatenate(dataset: Dataset[_], sets: Dataset[_]*): Dataset[_] = {
    var unionSet = dataset
    sets.foreach(ds => unionSet = unionSet.union(ds))

    class Record(val id: Long,
                 val label: Double,
                 val features: Vector,
                 val treeNum: Int,
                 val winId: Long)
    val concatData = unionSet.select($(instanceId), $(labelCol), $(featuresCol), $(treeNumCol), $(windowId)).rdd.map {
      row =>
        val id = row.getAs[Long]($(instanceId))
        val label = row.getAs[Double]($(labelCol))
        val features = row.getAs[Vector]($(featuresCol))
        val treeNum = row.getAs[Int]($(treeNumCol))
        val winId = row.getAs[Long]($(windowId))

        new Record(id, label, features, treeNum, winId)
    }.groupBy(
      record => record.id
    ).map { group =>
      val id = group._1
      val records = group._2
      val label = records.head.label

      def recordCompare(left: Record, right: Record): Boolean = {
        var code = left.treeNum.compareTo(right.treeNum)
        if (code == 0) code = left.winId.compareTo(right.winId)
        code <= 0
      }
      val features = new DenseVector(records.toSeq.sortWith(recordCompare).flatMap(_.features.toArray).toArray)
      val schema: StructType = StructType(Seq[StructField]())
        .add(StructField($(instanceId), LongType))
        .add(StructField($(labelCol), LongType))
        .add(StructField($(featuresCol), LongType))
      new GenericRowWithSchema(Array[Any](id, label, features), schema)
    }

    dataset.sparkSession.createDataset(concatData)
  }

  override protected def train(dataset: Dataset[_]): GCForestClassificationModel = {
    if ($(dataStyle) == "image") {
      require($(multiScanWindow).length % 2 == 0) // TODO comment
      val mgsModels = ArrayBuffer[MultiGrainedScanModel]()
      Range(0, $(multiScanWindow).length / 2).foreach { i =>
        val (w, h) = (i, i+1)
        val windowInstances = extractMatrixRDD(dataset, w, h)
        val rf = genRFClassifier("rf", $(scanForestTreeNum), $(scanForestMinInstancesPerNode))
        var (rfFeature, rfModel) = featureTransform(windowInstances, rf)
        rfFeature = rfFeature.withColumn($(treeNumCol), lit(1))

        val crtf = genRFClassifier("crtf", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode))
        var (crtfFeature, crtfModel) = featureTransform(windowInstances, crtf)
        crtfFeature = crtfFeature.withColumn($(treeNumCol), lit(2))
        mgsModels += new MultiGrainedScanModel(w, h, rfModel, crtfModel)

      }


    } else { // TODO
      throw new UnsupportedOperationException("Unsupported sequence data rightly!")
    }
  }

  override def copy(extra: ParamMap): GCForestClassifier = defaultCopy(extra)
}


class GCForestClassificationModel private[ml] ( override val uid: String,
                                                val scanWindow: Array[MultiGrainedScanModel],
                                                val cascadeForest: Array[EnsembleRandomForestModel]
                                           )
  extends ProbabilisticClassificationModel[Vector, GCForestClassificationModel]
    with MLWritable with Serializable {


}

class EnsembleRandomForestModel private[ml] (val forests: Array[RandomForestClassificationModel])

class MultiGrainedScanModel(val width: Int,
                       val height: Int,
                       val rfModel: RandomForestClassificationModel,
                       val crtfModel: RandomForestClassificationModel
                      ) {}