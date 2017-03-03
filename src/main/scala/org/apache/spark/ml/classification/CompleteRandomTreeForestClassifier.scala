package org.apache.spark.ml.classification

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.CompleteRandomTreeForest
import org.apache.spark.ml.util._
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

/**
  * Created by chengli on 3/3/17.
  */
class CompleteRandomTreeForestClassifier(override val uid: String) extends RandomForestClassifier {

  def this() = this(Identifiable.randomUID("crtfc"))

  override protected def train(dataset: Dataset[_]): RandomForestClassificationModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(params: _*)

    val trees = CompleteRandomTreeForest
      .run(oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeClassificationModel])

    val numFeatures = oldDataset.first().features.size
    val m = new RandomForestClassificationModel(trees, numFeatures, numClasses)
    instr.logSuccess(m)
    m
  }
}
