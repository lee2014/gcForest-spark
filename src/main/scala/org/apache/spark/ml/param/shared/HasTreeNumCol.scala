package org.apache.spark.ml.param.shared

import org.apache.spark.ml.param.{Param, Params}

/**
  * Created by chengli on 3/3/17.
  */
private[ml] trait HasTreeNumCol extends Params {
  final val treeNumCol: Param[String] = new Param[String](this, "treeNumCol", "tree number column name")

  setDefault(treeNumCol, "treeNum")

  /** @group getParam */
  final def getFeaturesCol: String = $(treeNumCol)
}


