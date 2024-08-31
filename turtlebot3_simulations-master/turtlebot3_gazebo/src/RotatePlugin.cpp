#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/Plugin.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo
{
  class RotatePlugin : public ModelPlugin
  {
    private: physics::ModelPtr model;
    private: event::ConnectionPtr updateConnection;

    public: void Load(physics::ModelPtr _model, sdf::ElementPtr /*_sdf*/)
    {
      // Store the model pointer for convenience.
      this->model = _model;

      gzdbg << "RotatePlugin loaded for model: " << _model->GetName() << std::endl;

      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&RotatePlugin::OnUpdate, this));
    }

    // Called by the world update start event
    public: void OnUpdate()
    {
      // Apply a small angular velocity around the Y axis.
      this->model->SetAngularVel(ignition::math::Vector3d(0, 1.0, 0));
      gzdbg << "Applying angular velocity" << std::endl;
    }
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(RotatePlugin)
}