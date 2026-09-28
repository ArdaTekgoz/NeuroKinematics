// C1-01 local IPC worker. Stdout is exclusively newline JSON protocol.
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/kinematics_base/kinematics_base.hpp>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <nlohmann/json.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

int64_t monotonic_ns() {
  // Matches Python time.monotonic_ns() on the supported Linux runtime.
  struct timespec value {};
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
    throw std::runtime_error("CLOCK_MONOTONIC unavailable");
  return static_cast<int64_t>(value.tv_sec) * 1000000000LL + value.tv_nsec;
}

std::string arg(int argc, char** argv, const std::string& key) {
  for (int i = 1; i + 1 < argc; ++i)
    if (argv[i] == key) return argv[i + 1];
  throw std::runtime_error("missing argument: " + key);
}

Json from_file(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open " + path);
  return Json::parse(file);
}

std::vector<double> finite_vector(const Json& value, size_t count, const char* name) {
  if (!value.is_array() || value.size() != count)
    throw std::runtime_error(std::string("invalid length: ") + name);
  std::vector<double> result;
  for (const auto& entry : value) {
    if (!entry.is_number() || !std::isfinite(entry.get<double>()))
      throw std::runtime_error(std::string("nonfinite ") + name);
    result.push_back(entry.get<double>());
  }
  return result;
}

void set_params(const rclcpp::Node::SharedPtr& node, const std::string& id) {
  const std::string prefix = "robot_description_kinematics.manipulator.";
  if (id == "kdl/default") {
    node->declare_parameter(prefix + "max_solver_iterations", 500);
    node->declare_parameter(prefix + "epsilon", 0.00001);
    node->declare_parameter(prefix + "orientation_vs_position", 1.0);
    node->declare_parameter(prefix + "position_only_ik", false);
  } else if (id == "trac_ik/speed") {
    node->declare_parameter(prefix + "epsilon", 0.00001);
    node->declare_parameter(prefix + "position_only_ik", false);
    node->declare_parameter(prefix + "solve_type", std::string("Speed"));
  } else if (id == "pick_ik/local" || id == "pick_ik/global") {
    node->declare_parameter(prefix + "mode", std::string(id == "pick_ik/local" ? "local" : "global"));
    node->declare_parameter(prefix + "position_threshold", 0.001);
    node->declare_parameter(prefix + "orientation_threshold", 0.008726646259971648);
    node->declare_parameter(prefix + "stop_optimization_on_valid_solution", true);
    node->declare_parameter(prefix + "minimal_displacement_weight", 0.0);
    node->declare_parameter(prefix + "memetic_num_threads", id == "pick_ik/local" ? 1 : 2);
  } else {
    throw std::runtime_error("unknown external solver: " + id);
  }
}

std::string plugin_name(const std::string& id) {
  if (id == "kdl/default") return "kdl_kinematics_plugin/KDLKinematicsPlugin";
  if (id == "trac_ik/speed") return "trac_ik_kinematics_plugin/TRAC_IKKinematicsPlugin";
  if (id == "pick_ik/local" || id == "pick_ik/global") return "pick_ik/PickIkPlugin";
  throw std::runtime_error("unknown external solver");
}

Json solve(const Json& request, const std::string& id, const std::string& hash,
           const std::vector<std::string>& joint_names, const Json& limits,
           const std::string& base, const std::string& tcp,
           const kinematics::KinematicsBase& plugin) {
  static const std::vector<std::string> fields = {
      "query_id", "q_current", "target_position_m", "target_quaternion_wxyz",
      "base_frame", "tcp_frame", "joint_order", "joint_limits", "deadline_ns",
      "solver_id", "solver_config_sha256", "seed", "expires_at_monotonic_ns"};
  if (!request.is_object() || request.size() != fields.size()) throw std::runtime_error("request fields mismatch");
  for (const auto& field : fields)
    if (!request.contains(field)) throw std::runtime_error("missing field: " + field);
  if (request.at("solver_id") != id || request.at("solver_config_sha256") != hash ||
      request.at("base_frame") != base || request.at("tcp_frame") != tcp ||
      request.at("joint_order") != joint_names || request.at("joint_limits") != limits)
    throw std::runtime_error("frozen solver/robot contract mismatch");
  if (!request.at("seed").is_null() && (!request.at("seed").is_number_integer() || request.at("seed").get<int64_t>() < 0))
    throw std::runtime_error("invalid seed");
  if (!request.at("deadline_ns").is_number_integer()) throw std::runtime_error("invalid deadline");
  int64_t deadline_ns = request.at("deadline_ns").get<int64_t>();
  if (deadline_ns != 10000000 && deadline_ns != 50000000) throw std::runtime_error("unfrozen deadline");
  const auto& expires_value = request.at("expires_at_monotonic_ns");
  if (!expires_value.is_number_integer() ||
      (expires_value.is_number_unsigned() &&
       expires_value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())))
    throw std::runtime_error("invalid monotonic expiry");
  const int64_t expires_ns = expires_value.get<int64_t>();
  if (expires_ns <= 0 || expires_ns - monotonic_ns() > deadline_ns)
    throw std::runtime_error("monotonic expiry exceeds frozen budget");
  auto seed = finite_vector(request.at("q_current"), joint_names.size(), "q_current");
  auto position = finite_vector(request.at("target_position_m"), 3, "target_position_m");
  auto quaternion = finite_vector(request.at("target_quaternion_wxyz"), 4, "target_quaternion_wxyz");
  double norm = 0.0;
  for (double v : quaternion) norm += v * v;
  if (std::abs(norm - 1.0) > 2e-8) throw std::runtime_error("target quaternion is not unit length");
  for (size_t i = 0; i < seed.size(); ++i)
    if (seed[i] < limits[i][0].get<double>() || seed[i] > limits[i][1].get<double>())
      throw std::runtime_error("seed outside frozen limits");

  geometry_msgs::msg::Pose pose;
  pose.position.x = position[0]; pose.position.y = position[1]; pose.position.z = position[2];
  pose.orientation.w = quaternion[0]; pose.orientation.x = quaternion[1];
  pose.orientation.y = quaternion[2]; pose.orientation.z = quaternion[3];
  std::vector<double> solution;
  moveit_msgs::msg::MoveItErrorCodes error;
  // IPC, parsing, validation and pose conversion consume the shared budget.
  // No relative full-budget reset is allowed on entry to a plugin.
  const int64_t remaining_ns = expires_ns - monotonic_ns();
  if (remaining_ns <= 0)
    return {{"query_id", request.at("query_id")}, {"solver_id", id},
            {"solver_config_sha256", hash}, {"native_status", "TIMEOUT"},
            {"termination_reason", "DEADLINE_EXPIRED_BEFORE_SOLVE"},
            {"q_candidate", nullptr}, {"iterations", nullptr},
            {"iteration_availability", "NOT_AVAILABLE"},
            {"solver_internal_elapsed_ns", 0}, {"error_class", nullptr}};
  auto started = Clock::now();
  bool found = plugin.searchPositionIK(pose, seed, remaining_ns / 1e9, solution, error);
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count();
  std::string status = found ? "SUCCESS" : (elapsed >= remaining_ns ? "TIMEOUT" : "UNRESOLVED");
  if (found && solution.size() != joint_names.size())
    throw std::runtime_error("plugin candidate has wrong joint count");
  Json candidate = nullptr;
  if (solution.size() == joint_names.size()) {
    for (double value : solution)
      if (!std::isfinite(value)) throw std::runtime_error("plugin candidate is nonfinite");
    candidate = solution;
  }
  return {{"query_id", request.at("query_id")}, {"solver_id", id},
          {"solver_config_sha256", hash}, {"native_status", status},
          {"termination_reason", "MOVEIT_ERROR_" + std::to_string(error.val)},
          {"q_candidate", candidate}, {"iterations", nullptr},
          {"iteration_availability", "NOT_AVAILABLE"},
          {"solver_internal_elapsed_ns", elapsed}, {"error_class", nullptr}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string id = arg(argc, argv, "--solver");
    const std::string hash = arg(argc, argv, "--config-hash");
    const std::string base = arg(argc, argv, "--base");
    const std::string tcp = arg(argc, argv, "--tcp");
    Json spec = from_file(arg(argc, argv, "--robot-spec"));
    const auto& joints = spec.at("mechanism").at("active_joints");
    std::vector<std::string> joint_names;
    Json limits = Json::array();
    for (const auto& joint : joints) {
      joint_names.push_back(joint.at("name").get<std::string>());
      limits.push_back({joint.at("limit").at("lower_rad"), joint.at("limit").at("upper_rad")});
    }
    if (joint_names != spec.at("mechanism").at("active_joint_order").get<std::vector<std::string>>())
      throw std::runtime_error("robot specification joint order mismatch");
    auto urdf = urdf::parseURDFFile(arg(argc, argv, "--urdf"));
    if (!urdf) throw std::runtime_error("URDF parse failed");
    const std::string srdf_xml = "<robot name=\"" + urdf->getName() +
                                 "\"><group name=\"manipulator\"><chain base_link=\"" + base +
                                 "\" tip_link=\"" + tcp + "\"/></group></robot>";
    auto srdf = std::make_shared<srdf::Model>();
    if (!srdf->initString(*urdf, srdf_xml)) throw std::runtime_error("SRDF parse failed");
    auto robot_model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    auto group = robot_model->getJointModelGroup("manipulator");
    if (!group || group->getActiveJointModelNames() != joint_names)
      throw std::runtime_error("MoveIt joint order differs from frozen robot specification");
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("c101_moveit_worker");
    set_params(node, id);
    pluginlib::ClassLoader<kinematics::KinematicsBase> loader("moveit_core", "kinematics::KinematicsBase");
    auto plugin = loader.createUniqueInstance(plugin_name(id));
    if (!plugin->initialize(node, *robot_model, "manipulator", base, {tcp}, 0.005) ||
        plugin->getJointNames() != joint_names || plugin->getBaseFrame() != base ||
        plugin->getTipFrame() != tcp)
      throw std::runtime_error("MoveIt plugin initialize/frame/joint contract failed");
    std::cout << Json({{"ready", true}, {"solver_id", id}, {"solver_config_sha256", hash}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
      try {
        std::cout << solve(Json::parse(line), id, hash, joint_names, limits, base, tcp, *plugin).dump() << std::endl;
      } catch (const std::exception& error) {
        std::cerr << "C1-01 worker rejected request: " << error.what() << std::endl;
        return 2;
      }
    }
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "C1-01 worker initialization failure: " << error.what() << std::endl;
    return 1;
  }
}
