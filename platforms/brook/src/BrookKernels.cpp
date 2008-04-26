/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008 Stanford University and the Authors.           *
 * Authors: Peter Eastman, Mark Friedrichs                                    *
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

#include "BrookKernels.h"
#include "BrookFloatStreamImpl.h"
#include "SimTKBrook/BrookAngleBondIxn.h"
#include "SimTKBrook/BrookBondForce.h"
#include "SimTKBrook/BrookHarmonicBondIxn.h"
#include "SimTKBrook/BrookLJCoulomb14.h"
#include "SimTKBrook/BrookLJCoulombIxn.h"
#include "SimTKBrook/BrookProperDihedralBond.h"
#include "SimTKBrook/BrookRbDihedralBond.h"
#include "SimTKBrook/BrookStochasticDynamics.h"
#include "SimTKBrook/BrookShakeAlgorithm.h"
#include <cmath>
#include <limits>

using namespace OpenMM;
using namespace std;

int** allocateIntArray(int length, int width) {
    int** array = new int*[length];
    for (int i = 0; i < length; ++i)
        array[i] = new int[width];
    return array;
}

RealOpenMM** allocateRealArray(int length, int width) {
    RealOpenMM** array = new RealOpenMM*[length];
    for (int i = 0; i < length; ++i)
        array[i] = new RealOpenMM[width];
    return array;
}

int** copyToArray(const vector<vector<int> > vec) {
    if (vec.size() == 0)
        return new int*[0];
    int** array = allocateIntArray(vec.size(), vec[0].size());
    for (int i = 0; i < vec.size(); ++i)
        for (int j = 0; j < vec[i].size(); ++j)
            array[i][j] = vec[i][j];
    return array;
}

RealOpenMM** copyToArray(const vector<vector<double> > vec) {
    if (vec.size() == 0)
        return new RealOpenMM*[0];
    RealOpenMM** array = allocateRealArray(vec.size(), vec[0].size());
    for (int i = 0; i < vec.size(); ++i)
        for (int j = 0; j < vec[i].size(); ++j)
            array[i][j] = vec[i][j];
    return array;
}

void disposeIntArray(int** array, int size) {
    if (array) {
        for (int i = 0; i < size; ++i)
            delete[] array[i];
        delete[] array;
    }
}

void disposeRealArray(RealOpenMM** array, int size) {
    if (array) {
        for (int i = 0; i < size; ++i)
            delete[] array[i];
        delete[] array;
    }
}

BrookCalcStandardMMForceFieldKernel::~BrookCalcStandardMMForceFieldKernel() {
    disposeIntArray(bondIndexArray, numBonds);
    disposeRealArray(bondParamArray, numBonds);
    disposeIntArray(angleIndexArray, numAngles);
    disposeRealArray(angleParamArray, numAngles);
    disposeIntArray(periodicTorsionIndexArray, numPeriodicTorsions);
    disposeRealArray(periodicTorsionParamArray, numPeriodicTorsions);
    disposeIntArray(rbTorsionIndexArray, numRBTorsions);
    disposeRealArray(rbTorsionParamArray, numRBTorsions);
    disposeRealArray(atomParamArray, numAtoms);
    disposeIntArray(exclusionArray, numAtoms);
    disposeIntArray(bonded14IndexArray, num14);
    disposeRealArray(bonded14ParamArray, num14);
}

void BrookCalcStandardMMForceFieldKernel::initialize(const vector<vector<int> >& bondIndices, const vector<vector<double> >& bondParameters,
        const vector<vector<int> >& angleIndices, const vector<vector<double> >& angleParameters,
        const vector<vector<int> >& periodicTorsionIndices, const vector<vector<double> >& periodicTorsionParameters,
        const vector<vector<int> >& rbTorsionIndices, const vector<vector<double> >& rbTorsionParameters,
        const vector<vector<int> >& bonded14Indices, double lj14Scale, double coulomb14Scale,
        const vector<set<int> >& exclusions, const vector<vector<double> >& nonbondedParameters) {
    numAtoms = nonbondedParameters.size();
    numBonds = bondIndices.size();
    numAngles = angleIndices.size();
    numPeriodicTorsions = periodicTorsionIndices.size();
    numRBTorsions = rbTorsionIndices.size();
    num14 = bonded14Indices.size();
    bondIndexArray = copyToArray(bondIndices);
    bondParamArray = copyToArray(bondParameters);
    angleIndexArray = copyToArray(angleIndices);
    angleParamArray = copyToArray(angleParameters);
    periodicTorsionIndexArray = copyToArray(periodicTorsionIndices);
    periodicTorsionParamArray = copyToArray(periodicTorsionParameters);
    rbTorsionIndexArray = copyToArray(rbTorsionIndices);
    rbTorsionParamArray = copyToArray(rbTorsionParameters);
    atomParamArray = allocateRealArray(numAtoms, 3);
    RealOpenMM sqrtEps = std::sqrt(138.935485);
    for (int i = 0; i < numAtoms; ++i) {
        atomParamArray[i][0] = 0.5*nonbondedParameters[i][1];
        atomParamArray[i][1] = 2.0*sqrt(nonbondedParameters[i][2]);
        atomParamArray[i][2] = nonbondedParameters[i][0]*sqrtEps;
    }
    exclusionArray = new int*[numAtoms];
    for (int i = 0; i < numAtoms; ++i) {
        exclusionArray[i] = new int[exclusions[i].size()+1];
        exclusionArray[i][0] = exclusions[i].size();
        int index = 0;
        for (set<int>::const_iterator iter = exclusions[i].begin(); iter != exclusions[i].end(); ++iter)
            exclusionArray[i][++index] = *iter;
    }
    bonded14IndexArray = copyToArray(bonded14Indices);
    bonded14ParamArray = allocateRealArray(num14, 3);
    for (int i = 0; i < num14; ++i) {
        int atom1 = bonded14Indices[i][0];
        int atom2 = bonded14Indices[i][1];
        bonded14ParamArray[i][0] = atomParamArray[atom1][0]+atomParamArray[atom2][0];
        bonded14ParamArray[i][1] = lj14Scale*(atomParamArray[atom1][1]*atomParamArray[atom2][1]);
        bonded14ParamArray[i][2] = coulomb14Scale*(atomParamArray[atom1][2]*atomParamArray[atom2][2]);
    }
}

void BrookCalcStandardMMForceFieldKernel::executeForces(const Stream& positions, Stream& forces) {
    RealOpenMM** posData = const_cast<RealOpenMM**>(((BrookFloatStreamImpl&) positions.getImpl()).getData()); // Brook code needs to be made const correct
    RealOpenMM** forceData = ((BrookFloatStreamImpl&) forces.getImpl()).getData();
    BrookBondForce refBondForce;
    BrookHarmonicBondIxn harmonicBond;
    refBondForce.calculateForce(numBonds, bondIndexArray, posData, bondParamArray, forceData, 0, 0, 0, harmonicBond);
    BrookAngleBondIxn angleBond;
    refBondForce.calculateForce(numAngles, angleIndexArray, posData, angleParamArray, forceData, 0, 0, 0, angleBond);
    BrookProperDihedralBond periodicTorsionBond;
    refBondForce.calculateForce(numPeriodicTorsions, periodicTorsionIndexArray, posData, periodicTorsionParamArray, forceData, 0, 0, 0, periodicTorsionBond);
    BrookRbDihedralBond rbTorsionBond;
    refBondForce.calculateForce(numRBTorsions, rbTorsionIndexArray, posData, rbTorsionParamArray, forceData, 0, 0, 0, rbTorsionBond);
    BrookLJCoulombIxn clj;
    clj.calculatePairIxn(numAtoms, posData, atomParamArray, exclusionArray, 0, forceData, 0, 0);
    BrookLJCoulomb14 nonbonded14;
    refBondForce.calculateForce(num14, bonded14IndexArray, posData, bonded14ParamArray, forceData, 0, 0, 0, nonbonded14);
}

double BrookCalcStandardMMForceFieldKernel::executeEnergy(const Stream& positions) {
    RealOpenMM** posData = const_cast<RealOpenMM**>(((BrookFloatStreamImpl&) positions.getImpl()).getData()); // Brook code needs to be made const correct
    RealOpenMM** forceData = allocateRealArray(numAtoms, 3);
    int arraySize = max(max(max(max(numAtoms, numBonds), numAngles), numPeriodicTorsions), numRBTorsions);
    RealOpenMM* energyArray = new RealOpenMM[arraySize];
    RealOpenMM energy = 0;
    BrookBondForce refBondForce;
    BrookHarmonicBondIxn harmonicBond;
    for (int i = 0; i < arraySize; ++i)
        energyArray[i] = 0;
    refBondForce.calculateForce(numBonds, bondIndexArray, posData, bondParamArray, forceData, energyArray, 0, &energy, harmonicBond);
    BrookAngleBondIxn angleBond;
    for (int i = 0; i < arraySize; ++i)
        energyArray[i] = 0;
    refBondForce.calculateForce(numAngles, angleIndexArray, posData, angleParamArray, forceData, energyArray, 0, &energy, angleBond);
    BrookProperDihedralBond periodicTorsionBond;
    for (int i = 0; i < arraySize; ++i)
        energyArray[i] = 0;
    refBondForce.calculateForce(numPeriodicTorsions, periodicTorsionIndexArray, posData, periodicTorsionParamArray, forceData, energyArray, 0, &energy, periodicTorsionBond);
    BrookRbDihedralBond rbTorsionBond;
    for (int i = 0; i < arraySize; ++i)
        energyArray[i] = 0;
    refBondForce.calculateForce(numRBTorsions, rbTorsionIndexArray, posData, rbTorsionParamArray, forceData, energyArray, 0, &energy, rbTorsionBond);
    BrookLJCoulombIxn clj;
    clj.calculatePairIxn(numAtoms, posData, atomParamArray, exclusionArray, 0, forceData, 0, &energy);
    BrookLJCoulomb14 nonbonded14;
    for (int i = 0; i < arraySize; ++i)
        energyArray[i] = 0;
    refBondForce.calculateForce(num14, bonded14IndexArray, posData, bonded14ParamArray, forceData, energyArray, 0, &energy, nonbonded14);
    disposeRealArray(forceData, numAtoms);
    delete[] energyArray;
    return energy;
}

void BrookCalcGBSAOBCForceFieldKernel::initialize(const vector<double>& bornRadii, const vector<vector<double> >& atomParameters,
        double solventDielectric, double soluteDielectric) {
    
}

void BrookCalcGBSAOBCForceFieldKernel::executeForces(const Stream& positions, Stream& forces) {
    
}

double BrookCalcGBSAOBCForceFieldKernel::executeEnergy(const Stream& positions) {
    return 0.0; // TODO implement correctly
}

void BrookIntegrateVerletStepKernel::initialize(const vector<double>& masses, const vector<vector<int> >& constraintIndices,
        const vector<double>& constraintLengths) {
    
}

void BrookIntegrateVerletStepKernel::execute(Stream& positions, Stream& velocities, const Stream& forces, double stepSize) {
    
}
#include <iostream>
BrookIntegrateLangevinStepKernel::~BrookIntegrateLangevinStepKernel() {
    if (dynamics)
        delete dynamics;
    if (shake)
        delete shake;
    if (masses)
        delete[] masses;
    if (constraintIndices)
        disposeIntArray(constraintIndices, numConstraints);
    if (shakeParameters)
        disposeRealArray(shakeParameters, numConstraints);
}

void BrookIntegrateLangevinStepKernel::initialize(const vector<double>& masses, const vector<vector<int> >& constraintIndices,
        const vector<double>& constraintLengths) {
    this->masses = new RealOpenMM[masses.size()];
    for (int i = 0; i < masses.size(); ++i)
        this->masses[i] = masses[i];
    numConstraints = constraintIndices.size();
    this->constraintIndices = allocateIntArray(numConstraints, 2);
    for (int i = 0; i < numConstraints; ++i) {
        this->constraintIndices[i][0] = constraintIndices[i][0];
        this->constraintIndices[i][1] = constraintIndices[i][1];
    }
    shakeParameters = allocateRealArray(constraintLengths.size(), 1);
    for (int i = 0; i < constraintLengths.size(); ++i)
        shakeParameters[i][0] = constraintLengths[i];
}

void BrookIntegrateLangevinStepKernel::execute(Stream& positions, Stream& velocities, const Stream& forces, double temperature, double friction, double stepSize) {
    RealOpenMM** posData = ((BrookFloatStreamImpl&) positions.getImpl()).getData();
    RealOpenMM** velData = ((BrookFloatStreamImpl&) velocities.getImpl()).getData();
    RealOpenMM** forceData = const_cast<RealOpenMM**>(((BrookFloatStreamImpl&) forces.getImpl()).getData()); // Brook code needs to be made const correct
    if (dynamics == 0 || temperature != prevTemp || friction != prevFriction || stepSize != prevStepSize) {
        // Recreate the computation objects with the new parameters.
        
        if (dynamics) {
            delete dynamics;
            delete shake;
        }
        RealOpenMM tau = (friction == 0.0 ? 0.0 : 1.0/friction);
        dynamics = new BrookStochasticDynamics(positions.getSize(), stepSize, tau, temperature);
        shake = new BrookShakeAlgorithm(numConstraints, constraintIndices, shakeParameters);
        dynamics->setBrookShakeAlgorithm(shake);
        prevTemp = temperature;
        prevFriction = friction;
        prevStepSize = stepSize;
    }
    dynamics->update(positions.getSize(), posData, velData, forceData, masses);
}

void BrookIntegrateBrownianStepKernel::initialize(const vector<double>& masses, const vector<vector<int> >& constraintIndices,
        const vector<double>& constraintLengths) {
    
}

void BrookIntegrateBrownianStepKernel::execute(Stream& positions, Stream& velocities, const Stream& forces, double temperature, double friction, double stepSize) {
    
}

void BrookApplyAndersenThermostatKernel::initialize(const vector<double>& masses) {
    
}

void BrookApplyAndersenThermostatKernel::execute(Stream& velocities, double temperature, double collisionFrequency, double stepSize) {
    
}

void BrookCalcKineticEnergyKernel::initialize(const vector<double>& masses) {
    this->masses = masses;
}

double BrookCalcKineticEnergyKernel::execute(const Stream& velocities) {
    RealOpenMM** velData = const_cast<RealOpenMM**>(((BrookFloatStreamImpl&) velocities.getImpl()).getData()); // Brook code needs to be made const correct
    double energy = 0.0;
    for (int i = 0; i < masses.size(); ++i)
        energy += masses[i]*(velData[i][0]*velData[i][0]+velData[i][1]*velData[i][1]+velData[i][2]*velData[i][2]);
    return 0.5*energy;
}
