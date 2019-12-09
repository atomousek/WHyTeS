#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

#include <ros/ros.h>
#include <whytemap_ros/WHyTeMapMsg.h>
#include <whytemap_ros/GetWHyTeMap.h>


namespace whytemap_ros {

struct WHyTeMapCluster {
  double weight{0.0};
  std::vector<double> centroid;
  std::vector<double> precision_matrix;

  WHyTeMapCluster(long degree);

  WHyTeMapCluster(const WHyTeMapClusterMsg &msg);

  WHyTeMapClusterMsg toROSMsg() const;

  /**
   * Get the precision matrix as an Eigen::MatrixXd.
   * @return The precision matrix as an Eigen Matrix.
   */
  inline Eigen::MatrixXd getPrecisionMatrix() const {
    std::vector<double> copy_precision_matrix = precision_matrix;
    // We can do this since we know centroid.size() = degree
    Eigen::MatrixXd copy_precision_matrix_eigen = Eigen::Map<Eigen::MatrixXd>(
        copy_precision_matrix.data(), centroid.size(), centroid.size());
    return copy_precision_matrix_eigen;
  }

  /**
   * Get the centroid as an Eigen::VectorXd.
   * @return The centroid as an Eigen Vector.
   */
  inline Eigen::VectorXd getCentroid() const {
    std::vector<double> copy_centroid = centroid;
    Eigen::VectorXd copy_centroid_eigen =
        Eigen::Map<Eigen::VectorXd>(copy_centroid.data(), centroid.size());
    return copy_centroid_eigen;
  }
};

class WHyTeMap {

public:
  WHyTeMap() {}
  ~WHyTeMap() = default;

  /**
   * Create a WHyTeMap object from an xml file.
   * @param fileName File name of XML file.
   */
  inline WHyTeMap(const std::string &fileName) { readFromXML(fileName); }

  /**
   * Create a WHyTeMap object from a WHyTeMap ROS message.
   */
  WHyTeMap(const WHyTeMapMsg &msg);

  WHyTeMapMsg toROSMsg() const;

  /**
   * Read a WHyTe-map xml and load it into the data structres.
   * @param fileName The file name for WHyTeMap xml.
   */
  void readFromXML(const std::string &fileName);

  /**
   * Obtain the likelihood value for a particular query.
   * @param time The time at which likelihood is to be computed.
   * @param x The x coordinate
   * @param y The y coordinate
   * @param heading The heading
   * @param speed The speed.
   * @return Likelihood of a certain instance of position and velocity at a
   * certain time.
   */
  double getCost(double time, double x, double y, double heading,
                       double speed) const;

  inline void setFrameID(const std::string &frame_id) {
    this->frame_id_ = frame_id;
  }

  inline std::string getFrameID() const { return frame_id_; }
  inline std::vector<WHyTeMapCluster> getClusters() const { return clusters_; }
  inline std::vector<double> getPeriods() const { return periods_; }
  inline long getNoPeriods() const { return no_periods_; }
  inline long getNoClusters() const { return no_clusters_; }
  inline long getSpatialDim() const { return spatial_dim_; }

private:
  /// No of clusters in the WHyTeMap
  long no_clusters_;

  /// The periodicity
  long no_periods_;

  /// What's the dimension of the random variable?
  /// This is the number of spatial dimensions + 2 * periodicity
  long degree_;

  /// Number of spatial dimensions. It's 4 now. 2D position and 2D velocity
  long spatial_dim_{4};

  /// Chosen periodicity for hypertime projection
  std::vector<double> periods_;

  /// Clusters in the WHyTeMap
  std::vector<WHyTeMapCluster> clusters_;

  /// ROS frame id;
  std::string frame_id_;

  std::vector<std::vector<double>> grid_weights = {
    {0.000000000000000000e+00, 2.479338842975206680e-02, 4.628099173553719137e-01, 8.264462809917355601e-01, 9.090909090909090606e-01, 8.264462809917355601e-01, 4.628099173553719137e-01, 2.479338842975206680e-02, 0.000000000000000000e+00},
    {2.479338842975206680e-02, 7.520661157024793875e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 7.520661157024793875e-01, 2.479338842975206680e-02},
    {4.628099173553719137e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 4.628099173553719137e-01},
    {8.264462809917355601e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 8.264462809917355601e-01},
    {9.090909090909090606e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 9.090909090909090606e-01},
    {8.264462809917355601e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 8.264462809917355601e-01},
    {4.628099173553719137e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 4.628099173553719137e-01},
    {2.479338842975206680e-02, 7.520661157024793875e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 7.520661157024793875e-01, 2.479338842975206680e-02},
    {0.000000000000000000e+00, 2.479338842975206680e-02, 4.628099173553719137e-01, 8.264462809917355601e-01, 9.090909090909090606e-01, 8.264462809917355601e-01, 4.628099173553719137e-01, 2.479338842975206680e-02, 0.000000000000000000e+00}
  };
  
};

class WHyTeMapClient {
  ros::NodeHandle nh;
  ros::ServiceClient whytemap_client;

public:
  WHyTeMapClient(const std::string &service_name = "/get_cliffmap");

  WHyTeMapMsg get();
};

} // namespace whytemap_ros

std::ostream &operator<<(std::ostream &out,
                         const whytemap_ros::WHyTeMapCluster &clus);
std::ostream &operator<<(std::ostream &out, const whytemap_ros::WHyTeMap &map);
