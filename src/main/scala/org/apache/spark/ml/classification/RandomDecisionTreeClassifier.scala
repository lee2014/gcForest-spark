package org.apache.spark.ml.classification

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.CompleteRandomTreeForest
import org.apache.spark.ml.util.{Identifiable, Instrumentation, MetadataUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

/**
  * Created by chengli on 3/3/17.
  */
class RandomDecisionTreeClassifier(override val uid: String) extends DecisionTreeClassifier {

  def this() = this(Identifiable.randomUID("rdtc"))

  override protected def train(dataset: Dataset[_]): DecisionTreeClassificationModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy = getOldStrategy(categoricalFeatures, numClasses)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(params: _*)

    val trees = CompleteRandomTreeForest.run(oldDataset, strategy, numTrees = 1, featureSubsetStrategy = "all",
      seed = $(seed), instr = Some(instr), parentUID = Some(uid))

    val m = trees.head.asInstanceOf[DecisionTreeClassificationModel]
    instr.logSuccess(m)
    m
  }
}
