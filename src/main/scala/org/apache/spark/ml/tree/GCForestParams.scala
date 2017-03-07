package org.apache.spark.ml.tree

import org.apache.spark.ml.param.{IntArrayParam, IntParam, Param}
import org.apache.spark.ml.param.shared.{HasSeed, HasTreeNumCol}

/**
  * Created by chengli on 3/4/17.
  */
private[ml] trait GCForestParams extends HasSeed with HasTreeNumCol {

  final val multiScanWindow: IntArrayParam = new IntArrayParam(
    this, "multiScanWindow", "", (value: Array[Int]) => value.length > 0)

  setDefault(multiScanWindow -> Array[Int](100, 100))
  def setMultiScanWindow(value: Array[Int]): this.type = set(multiScanWindow, value)

  final val scanForestTreeNum: IntParam = new IntParam(
    this, "scanForestTreeNum", "", (value: Int) => value > 0)

  setDefault(scanForestTreeNum -> 30)
  def setScanForestTreeNum(value: Int): this.type = set(scanForestTreeNum, value)

  final val scanForestMinInstancesPerNode: IntParam = new IntParam(
    this, "scanForestMinInstancesPerNode", "", (value: Int) => value > 0)

  setDefault(scanForestMinInstancesPerNode -> 20)
  def setScanForestMinInstancesPerNode(value: Int): this.type =
    set(scanForestMinInstancesPerNode, value)

  final val cascadeForestMaxIteration: IntParam = new IntParam(
    this, "cascadeForestMaxIteration", "", (value: Int) => value > 0)

  setDefault(cascadeForestMaxIteration -> 50)
  def setMaxIteration(value: Int): this.type = set(cascadeForestMaxIteration, value)

  final val cascadeForestTreeNum: IntParam = new IntParam(
    this, "cascadeForestTreeNum", "", (value: Int) => value > 0)

  setDefault(cascadeForestTreeNum -> 1000)
  def setCascadeForestTreeNum(value: Int): this.type = set(cascadeForestTreeNum, value)

  final val cascadeForestMinInstancesPerNode: IntParam = new IntParam(
    this, "cascadeForestMinInstancesPerNode", "", (value: Int) => value > 0)

  setDefault(cascadeForestMinInstancesPerNode -> 10)
  def setCascadeForestMinInstancesPerNode(value: Int): this.type =
    set(cascadeForestMinInstancesPerNode, value)

  final val numFolds: IntParam = new IntParam(this, "numFolds", "", (value: Int) => value > 0) // TODO

  setDefault(numFolds -> 3)
  def setNumFolds(value: Int): this.type = set(numFolds, value)

  final val dataStyle: Param[String] = new Param[String](
    this, "dataStyle", "", (value: String) => Array("sequence", "image").contains(value)) // TODO

  setDefault(dataStyle -> "image")
  def setDataStyle(value: String): this.type = set(dataStyle, value)

  val dataSize: IntArrayParam = new IntArrayParam(this, "dataSize", "") // TODO

  def setDataSize(value: Array[Int]): this.type = set(dataSize, value)
  def getDataSize: Array[Int] = $(dataSize)

  final val instanceCol: Param[String] = new Param[String](this, "instanceCol", "instanceId column name")
  setDefault(instanceCol -> "instance")

  final val windowCol: Param[String] = new Param[String](this, "windowCol", "windowId column name")
  setDefault(windowCol -> "window")

  final val scanCol: Param[String] = new Param[String](this, "scanCol", "scanId column name")
  setDefault(scanCol -> "scan_id")

  def setSeed(value: Long): this.type = set(seed, value)
}
