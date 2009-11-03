#ifndef OPENMM_KERNELS_H_
#define OPENMM_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2009 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/AndersenThermostat.h"
#include "openmm/BrownianIntegrator.h"
#include "openmm/CMMotionRemover.h"
#include "openmm/CustomNonbondedForce.h"
#include "openmm/GBSAOBCForce.h"
#include "openmm/GBVIForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/LangevinIntegrator.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/RBTorsionForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VariableLangevinIntegrator.h"
#include "openmm/VariableVerletIntegrator.h"
#include "openmm/VerletIntegrator.h"
#include <set>
#include <string>
#include <vector>

namespace OpenMM {

/**
 * This kernel is invoked at the beginning and end of force and energy computations.  It gives the
 * Platform a chance to clear buffers and do other initialization at the beginning, and to do any
 * necessary work at the end to determine the final results.
 */
class CalcForcesAndEnergyKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcForcesAndEnergyKernel";
    }
    CalcForcesAndEnergyKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     */
    virtual void initialize(const System& system) = 0;
    /**
     * This is called at the beginning of each force computation, before calcForces() has been called on
     * any ForceImpl.
     * 
     * @param context    the context in which to execute this kernel
     */
    virtual void beginForceComputation(ContextImpl& context) = 0;
    /**
     * This is called at the end of each force computation, after calcForces() has been called on
     * every ForceImpl.
     *
     * @param context    the context in which to execute this kernel
     */
    virtual void finishForceComputation(ContextImpl& context) = 0;
    /**
     * This is called at the beginning of each energy computation, before calcEnergy() has been called on
     * any ForceImpl.
     *
     * @param context    the context in which to execute this kernel
     */
    virtual void beginEnergyComputation(ContextImpl& context) = 0;
    /**
     * This is called at the end of each energy computation, after calcEnergy() has been called on
     * every ForceImpl.
     *
     * @param context    the context in which to execute this kernel
     * @return the potential energy of the system.  This value is added to all values returned by ForceImpls'
     * calcEnergy() methods.  That is, each force kernel may <i>either</i> return its contribution to the
     * energy directly, <i>or</i> add it to an internal buffer so that it will be included here.
     */
    virtual double finishEnergyComputation(ContextImpl& context) = 0;
};

/**
 * This kernel provides methods for setting and retrieving various state data: time, positions,
 * velocities, and forces.
 */
class UpdateStateDataKernel : public KernelImpl {
public:
    static std::string Name() {
        return "UpdateTime";
    }
    UpdateStateDataKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     */
    virtual void initialize(const System& system) = 0;
    /**
     * Get the current time (in picoseconds).
     *
     * @param context    the context in which to execute this kernel
     */
    virtual double getTime(const ContextImpl& context) const = 0;
    /**
     * Set the current time (in picoseconds).
     *
     * @param context    the context in which to execute this kernel
     * @param time       the time
     */
    virtual void setTime(ContextImpl& context, double time) = 0;
    /**
     * Get the positions of all particles.
     *
     * @param positions  on exit, this contains the particle positions
     */
    virtual void getPositions(ContextImpl& context, std::vector<Vec3>& positions) = 0;
    /**
     * Set the positions of all particles.
     *
     * @param positions  a vector containg the particle positions
     */
    virtual void setPositions(ContextImpl& context, const std::vector<Vec3>& positions) = 0;
    /**
     * Get the velocities of all particles.
     *
     * @param velocities  on exit, this contains the particle velocities
     */
    virtual void getVelocities(ContextImpl& context, std::vector<Vec3>& velocities) = 0;
    /**
     * Set the velocities of all particles.
     *
     * @param velocities  a vector containg the particle velocities
     */
    virtual void setVelocities(ContextImpl& context, const std::vector<Vec3>& velocities) = 0;
    /**
     * Get the current forces on all particles.
     *
     * @param forces  on exit, this contains the forces
     */
    virtual void getForces(ContextImpl& context, std::vector<Vec3>& forces) = 0;
};

/**
 * This kernel is invoked by HarmonicBondForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcHarmonicBondForceKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcHarmonicBondForce";
    }
    CalcHarmonicBondForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the HarmonicBondForce this kernel will be used for
     */
    virtual void initialize(const System& system, const HarmonicBondForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces.
     * 
     * @param context    the context in which to execute this kernel
     */
    virtual void executeForces(ContextImpl& context) = 0;
    /**
     * Execute the kernel to calculate the energy.
     * 
     * @param context    the context in which to execute this kernel
     * @return the potential energy due to the HarmonicBondForce
     */
    virtual double executeEnergy(ContextImpl& context) = 0;
};

/**
 * This kernel is invoked by HarmonicAngleForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcHarmonicAngleForceKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcHarmonicAngleForce";
    }
    CalcHarmonicAngleForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the HarmonicAngleForce this kernel will be used for
     */
    virtual void initialize(const System& system, const HarmonicAngleForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces.
     * 
     * @param context    the context in which to execute this kernel
     */
    virtual void executeForces(ContextImpl& context) = 0;
    /**
     * Execute the kernel to calculate the energy.
     * 
     * @param context    the context in which to execute this kernel
     * @return the potential energy due to the HarmonicAngleForce
     */
    virtual double executeEnergy(ContextImpl& context) = 0;
};

/**
 * This kernel is invoked by PeriodicTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcPeriodicTorsionForceKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcPeriodicTorsionForce";
    }
    CalcPeriodicTorsionForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the PeriodicTorsionForce this kernel will be used for
     */
    virtual void initialize(const System& system, const PeriodicTorsionForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces.
     * 
     * @param context    the context in which to execute this kernel
     */
    virtual void executeForces(ContextImpl& context) = 0;
    /**
     * Execute the kernel to calculate the energy.
     * 
     * @param context    the context in which to execute this kernel
     * @return the potential energy due to the PeriodicTorsionForce
     */
    virtual double executeEnergy(ContextImpl& context) = 0;
};

/**
 * This kernel is invoked by RBTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcRBTorsionForceKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcRBTorsionForce";
    }
    CalcRBTorsionForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the RBTorsionForce this kernel will be used for
     */
    virtual void initialize(const System& system, const RBTorsionForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces.
     * 
     * @param context    the context in which to execute this kernel
     */
    virtual void executeForces(ContextImpl& context) = 0;
    /**
     * Execute the kernel to calculate the energy.
     * 
     * @param context    the context in which to execute this kernel
     * @return the potential energy due to the RBTorsionForce
     */
    virtual double executeEnergy(ContextImpl& context) = 0;
};

/**
 * This kernel is invoked by NonbondedForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcNonbondedForceKernel : public KernelImpl {
public:
    enum NonbondedMethod {
        NoCutoff = 0,
        CutoffNonPeriodic = 1,
        CutoffPeriodic = 2,
        Ewald = 3,
        PME = 4
    };
    static std::string Name() {
        return "CalcNonbondedForce";
    }
    CalcNonbondedForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the NonbondedForce this kernel will be used for
     */
    virtual void initialize(const System& system, const NonbondedForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces.
     * 
     * @param context    the context in which to execute this kernel
     */
    virtual void executeForces(ContextImpl& context) = 0;
    /**
     * Execute the kernel to calculate the energy.
     * 
     * @param context    the context in which to execute this kernel
     * @return the potential energy due to the NonbondedForce
     */
    virtual double executeEnergy(ContextImpl& context) = 0;
};

/**
 * This kernel is invoked by CustomNonbondedForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcCustomNonbondedForceKernel : public KernelImpl {
public:
    enum NonbondedMethod {
        NoCutoff = 0,
        CutoffNonPeriodic = 1,
        CutoffPeriodic = 2
    };
    static std::string Name() {
        return "CalcCustomNonbondedForce";
    }
    CalcCustomNonbondedForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomNonbondedForce this kernel will be used for
     */
    virtual void initialize(const System& system, const CustomNonbondedForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces.
     *
     * @param context    the context in which to execute this kernel
     */
    virtual void executeForces(ContextImpl& context) = 0;
    /**
     * Execute the kernel to calculate the energy.
     *
     * @param context    the context in which to execute this kernel
     * @return the potential energy due to the CustomNonbondedForce
     */
    virtual double executeEnergy(ContextImpl& context) = 0;
};

/**
 * This kernel is invoked by GBSAOBCForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcGBSAOBCForceKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcGBSAOBCForce";
    }
    CalcGBSAOBCForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the GBSAOBCForce this kernel will be used for
     */
    virtual void initialize(const System& system, const GBSAOBCForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces.
     * 
     * @param context    the context in which to execute this kernel
     */
    virtual void executeForces(ContextImpl& context) = 0;
    /**
     * Execute the kernel to calculate the energy.
     * 
     * @param context    the context in which to execute this kernel
     * @return the potential energy due to the GBSAOBCForce
     */
    virtual double executeEnergy(ContextImpl& context) = 0;
};

/**
 * This kernel is invoked by GBVIForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcGBVIForceKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcGBVIForce";
    }
    CalcGBVIForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system      the System this kernel will be applied to
     * @param force       the GBVIForce this kernel will be used for
     * @param scaledRadii scaled radii
     */
    virtual void initialize(const System& system, const GBVIForce& force, const std::vector<double>& scaledRadii) = 0;
    /**
     * Execute the kernel to calculate the forces.
     * 
     * @param context    the context in which to execute this kernel
     */
    virtual void executeForces(ContextImpl& context) = 0;
    /**
     * Execute the kernel to calculate the energy.
     * 
     * @param context    the context in which to execute this kernel
     * @return the potential energy due to the GBVIForce
     */
    virtual double executeEnergy(ContextImpl& context) = 0;
};

/**
 * This kernel is invoked by VerletIntegrator to take one time step.
 */
class IntegrateVerletStepKernel : public KernelImpl {
public:
    static std::string Name() {
        return "IntegrateVerletStep";
    }
    IntegrateVerletStepKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param integrator the VerletIntegrator this kernel will be used for
     */
    virtual void initialize(const System& system, const VerletIntegrator& integrator) = 0;
    /**
     * Execute the kernel.
     * 
     * @param context    the context in which to execute this kernel
     * @param integrator the VerletIntegrator this kernel is being used for
     */
    virtual void execute(ContextImpl& context, const VerletIntegrator& integrator) = 0;
};

/**
 * This kernel is invoked by LangevinIntegrator to take one time step.
 */
class IntegrateLangevinStepKernel : public KernelImpl {
public:
    static std::string Name() {
        return "IntegrateLangevinStep";
    }
    IntegrateLangevinStepKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param integrator the LangevinIntegrator this kernel will be used for
     */
    virtual void initialize(const System& system, const LangevinIntegrator& integrator) = 0;
    /**
     * Execute the kernel.
     * 
     * @param context    the context in which to execute this kernel
     * @param integrator the LangevinIntegrator this kernel is being used for
     */
    virtual void execute(ContextImpl& context, const LangevinIntegrator& integrator) = 0;
};

/**
 * This kernel is invoked by BrownianIntegrator to take one time step.
 */
class IntegrateBrownianStepKernel : public KernelImpl {
public:
    static std::string Name() {
        return "IntegrateBrownianStep";
    }
    IntegrateBrownianStepKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param integrator the BrownianIntegrator this kernel will be used for
     */
    virtual void initialize(const System& system, const BrownianIntegrator& integrator) = 0;
    /**
     * Execute the kernel.
     * 
     * @param context    the context in which to execute this kernel
     * @param integrator the BrownianIntegrator this kernel is being used for
     */
    virtual void execute(ContextImpl& context, const BrownianIntegrator& integrator) = 0;
};

/**
 * This kernel is invoked by VariableLangevinIntegrator to take one time step.
 */
class IntegrateVariableLangevinStepKernel : public KernelImpl {
public:
    static std::string Name() {
        return "IntegrateVariableLangevinStep";
    }
    IntegrateVariableLangevinStepKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the VariableLangevinIntegrator this kernel will be used for
     */
    virtual void initialize(const System& system, const VariableLangevinIntegrator& integrator) = 0;
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the LangevinIntegrator this kernel is being used for
     * @param maxTime    the maximum time beyond which the simulation should not be advanced
     */
    virtual void execute(ContextImpl& context, const VariableLangevinIntegrator& integrator, double maxTime) = 0;
};

/**
 * This kernel is invoked by VariableVerletIntegrator to take one time step.
 */
class IntegrateVariableVerletStepKernel : public KernelImpl {
public:
    static std::string Name() {
        return "IntegrateVariableVerletStep";
    }
    IntegrateVariableVerletStepKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the VariableVerletIntegrator this kernel will be used for
     */
    virtual void initialize(const System& system, const VariableVerletIntegrator& integrator) = 0;
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the VerletIntegrator this kernel is being used for
     * @param maxTime    the maximum time beyond which the simulation should not be advanced
     */
    virtual void execute(ContextImpl& context, const VariableVerletIntegrator& integrator, double maxTime) = 0;
};

/**
 * This kernel is invoked by AndersenThermostat at the start of each time step to adjust the particle velocities.
 */
class ApplyAndersenThermostatKernel : public KernelImpl {
public:
    static std::string Name() {
        return "ApplyAndersenThermostat";
    }
    ApplyAndersenThermostatKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param thermostat the AndersenThermostat this kernel will be used for
     */
    virtual void initialize(const System& system, const AndersenThermostat& thermostat) = 0;
    /**
     * Execute the kernel.
     * 
     * @param context    the context in which to execute this kernel
     */
    virtual void execute(ContextImpl& context) = 0;
};

/**
 * This kernel is invoked to calculate the kinetic energy of the system.
 */
class CalcKineticEnergyKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcKineticEnergy";
    }
    CalcKineticEnergyKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     */
    virtual void initialize(const System& system) = 0;
    /**
     * Execute the kernel.
     * 
     * @param context    the context in which to execute this kernel
     */
    virtual double execute(ContextImpl& context) = 0;
};

/**
 * This kernel is invoked to remove center of mass motion from the system.
 */
class RemoveCMMotionKernel : public KernelImpl {
public:
    static std::string Name() {
        return "RemoveCMMotion";
    }
    RemoveCMMotionKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the CMMotionRemover this kernel will be used for
     */
    virtual void initialize(const System& system, const CMMotionRemover& force) = 0;
    /**
     * Execute the kernel.
     * 
     * @param context    the context in which to execute this kernel
     */
    virtual void execute(ContextImpl& context) = 0;
};

} // namespace OpenMM

#endif /*OPENMM_KERNELS_H_*/
