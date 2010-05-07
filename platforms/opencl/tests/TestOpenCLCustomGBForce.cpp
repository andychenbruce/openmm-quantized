
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

/**
 * This tests the OpenCL implementation of CustomGBForce.
 */

#include "../../../tests/AssertionUtilities.h"
#include "../src/sfmt/SFMT.h"
#include "openmm/Context.h"
#include "OpenCLPlatform.h"
#include "openmm/CustomGBForce.h"
#include "openmm/GBSAOBCForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;

void testOBC(GBSAOBCForce::NonbondedMethod obcMethod, CustomGBForce::NonbondedMethod customMethod) {
    const int numMolecules = 70;
    const int numParticles = numMolecules*2;
    const double boxSize = 10.0;
    OpenCLPlatform platform;

    // Create two systems: one with a GBSAOBCForce, and one using a CustomGBForce to implement the same interaction.

    System standardSystem;
    System customSystem;
    for (int i = 0; i < numParticles; i++) {
        standardSystem.addParticle(1.0);
        customSystem.addParticle(1.0);
    }
    standardSystem.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0.0, 0.0), Vec3(0.0, boxSize, 0.0), Vec3(0.0, 0.0, boxSize));
    customSystem.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0.0, 0.0), Vec3(0.0, boxSize, 0.0), Vec3(0.0, 0.0, boxSize));
    GBSAOBCForce* obc = new GBSAOBCForce();
    CustomGBForce* custom = new CustomGBForce();
    obc->setCutoffDistance(2.0);
    custom->setCutoffDistance(2.0);
    custom->addPerParticleParameter("q");
    custom->addPerParticleParameter("radius");
    custom->addPerParticleParameter("scale");
    custom->addGlobalParameter("solventDielectric", obc->getSolventDielectric());
    custom->addGlobalParameter("soluteDielectric", obc->getSoluteDielectric());
    custom->addComputedValue("I", "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
                                  "U=r+sr2;"
                                  "C=2*(1/or1-1/L)*step(sr2-r-or1);"
                                  "L=max(or1, D);"
                                  "D=abs(r-sr2);"
                                  "sr2 = scale2*or2;"
                                  "or1 = radius1-0.009; or2 = radius2-0.009", CustomGBForce::ParticlePairNoExclusions);
    custom->addComputedValue("B", "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
                                  "psi=I*or; or=radius-0.009", CustomGBForce::SingleParticle);
    custom->addEnergyTerm("28.3919551*(radius+0.14)^2*(radius/B)^6-0.5*138.935456*(1/soluteDielectric-1/solventDielectric)*q^2/B", CustomGBForce::SingleParticle);
    custom->addEnergyTerm("-138.935456*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;"
                          "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", CustomGBForce::ParticlePairNoExclusions);
    vector<Vec3> positions(numParticles);
    vector<Vec3> velocities(numParticles);
    init_gen_rand(0);
    vector<double> params(3);
    for (int i = 0; i < numMolecules; i++) {
        if (i < numMolecules/2) {
            obc->addParticle(1.0, 0.2, 0.5);
            params[0] = 1.0;
            params[1] = 0.2;
            params[2] = 0.5;
            custom->addParticle(params);
            obc->addParticle(-1.0, 0.1, 0.5);
            params[0] = -1.0;
            params[1] = 0.1;
            custom->addParticle(params);
        }
        else {
            obc->addParticle(1.0, 0.2, 0.8);
            params[0] = 1.0;
            params[1] = 0.2;
            params[2] = 0.8;
            custom->addParticle(params);
            obc->addParticle(-1.0, 0.1, 0.8);
            params[0] = -1.0;
            params[1] = 0.1;
            custom->addParticle(params);
        }
        positions[2*i] = Vec3(boxSize*genrand_real2(), boxSize*genrand_real2(), boxSize*genrand_real2());
        positions[2*i+1] = Vec3(positions[2*i][0]+1.0, positions[2*i][1], positions[2*i][2]);
        velocities[2*i] = Vec3(genrand_real2(), genrand_real2(), genrand_real2());
        velocities[2*i+1] = Vec3(genrand_real2(), genrand_real2(), genrand_real2());
    }
    obc->setNonbondedMethod(obcMethod);
    custom->setNonbondedMethod(customMethod);
    standardSystem.addForce(obc);
    customSystem.addForce(custom);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context context1(standardSystem, integrator1, platform);
    context1.setPositions(positions);
    context1.setVelocities(velocities);
    State state1 = context1.getState(State::Forces | State::Energy);
    Context context2(customSystem, integrator2, platform);
    context2.setPositions(positions);
    context2.setVelocities(velocities);
    State state2 = context2.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(state1.getPotentialEnergy(), state2.getPotentialEnergy(), 1e-4);
    for (int i = 0; i < numParticles; i++) {
        ASSERT_EQUAL_VEC(state1.getForces()[i], state2.getForces()[i], 1e-4);
    }
}

void testTabulatedFunction(bool interpolating) {
    OpenCLPlatform platform;
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    CustomGBForce* force = new CustomGBForce();
    force->addComputedValue("a", "0", CustomGBForce::ParticlePair);
    force->addEnergyTerm("fn(r)+1", CustomGBForce::ParticlePair);
    force->addParticle(vector<double>());
    force->addParticle(vector<double>());
    vector<double> table;
    for (int i = 0; i < 21; i++)
        table.push_back(std::sin(0.25*i));
    force->addFunction("fn", table, 1.0, 6.0, interpolating);
    system.addForce(force);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    for (int i = 1; i < 30; i++) {
        double x = (7.0/30.0)*i;
        positions[1] = Vec3(x, 0, 0);
        context.setPositions(positions);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces = state.getForces();
        double force = (x < 1.0 || x > 6.0 ? 0.0 : -std::cos(x-1.0));
        double energy = (x < 1.0 || x > 6.0 ? 0.0 : std::sin(x-1.0))+1.0;
        ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[0], 0.1);
        ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[1], 0.1);
        ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), 0.02);
    }
}

void testMultipleChainRules() {
    OpenCLPlatform platform;
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    CustomGBForce* force = new CustomGBForce();
    force->addComputedValue("a", "2*r", CustomGBForce::ParticlePair);
    force->addComputedValue("b", "a+1", CustomGBForce::SingleParticle);
    force->addComputedValue("c", "2*b+a", CustomGBForce::SingleParticle);
    force->addEnergyTerm("0.1*a+1*b+10*c", CustomGBForce::SingleParticle); // 0.1*(2*r) + 2*r+1 + 10*(3*a+2) = 0.2*r + 2*r+1 + 40*r+20+20*r = 62.2*r+21
    force->addParticle(vector<double>());
    force->addParticle(vector<double>());
    system.addForce(force);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    for (int i = 1; i < 5; i++) {
        positions[1] = Vec3(i, 0, 0);
        context.setPositions(positions);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces = state.getForces();
        ASSERT_EQUAL_VEC(Vec3(124.4, 0, 0), forces[0], 1e-4);
        ASSERT_EQUAL_VEC(Vec3(-124.4, 0, 0), forces[1], 1e-4);
        ASSERT_EQUAL_TOL(2*(62.2*i+21), state.getPotentialEnergy(), 0.02);
    }
}

void testPositionDependence() {
    OpenCLPlatform platform;
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    CustomGBForce* force = new CustomGBForce();
    force->addComputedValue("a", "r", CustomGBForce::ParticlePair);
    force->addComputedValue("b", "a+y", CustomGBForce::SingleParticle);
    force->addEnergyTerm("b*z", CustomGBForce::SingleParticle);
    force->addEnergyTerm("b1+b2", CustomGBForce::ParticlePair); // = 2*r+y1+y2
    force->addParticle(vector<double>());
    force->addParticle(vector<double>());
    system.addForce(force);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    vector<Vec3> forces(2);
    init_gen_rand(0);
    for (int i = 0; i < 5; i++) {
        positions[0] = Vec3(genrand_real2(), genrand_real2(), genrand_real2());
        positions[1] = Vec3(genrand_real2(), genrand_real2(), genrand_real2());
        context.setPositions(positions);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces = state.getForces();
        Vec3 delta = positions[0]-positions[1];
        double r = sqrt(delta.dot(delta));
        double energy = 2*r+positions[0][1]+positions[1][1];
        for (int j = 0; j < 2; j++)
            energy += positions[j][2]*(r+positions[j][1]);
        Vec3 force1(-(1+positions[0][2])*delta[0]/r-(1+positions[1][2])*delta[0]/r,
                    -(1+positions[0][2])*delta[1]/r-positions[0][2]-(1+positions[1][2])*delta[1]/r-1,
                    -(1+positions[0][2])*delta[2]/r-(r+positions[0][1])-(1+positions[1][2])*delta[2]/r);
        Vec3 force2((1+positions[0][2])*delta[0]/r+(1+positions[1][2])*delta[0]/r,
                    (1+positions[0][2])*delta[1]/r+(1+positions[1][2])*delta[1]/r-positions[1][2]-1,
                    (1+positions[0][2])*delta[2]/r+(1+positions[1][2])*delta[2]/r-(r+positions[1][1]));
        ASSERT_EQUAL_VEC(force1, forces[0], 1e-4);
        ASSERT_EQUAL_VEC(force2, forces[1], 1e-4);
        ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), 0.02);

        // Take a small step in the direction of the energy gradient and see whether the potential energy changes by the expected amount.

        double norm = 0.0;
        for (int i = 0; i < (int) forces.size(); ++i)
            norm += forces[i].dot(forces[i]);
        norm = std::sqrt(norm);
        const double stepSize = 1e-3;
        double step = stepSize/norm;
        for (int i = 0; i < (int) positions.size(); ++i) {
            Vec3 p = positions[i];
            Vec3 f = forces[i];
            positions[i] = Vec3(p[0]-f[0]*step, p[1]-f[1]*step, p[2]-f[2]*step);
        }
        context.setPositions(positions);
        State state2 = context.getState(State::Energy);
        ASSERT_EQUAL_TOL(norm, (state2.getPotentialEnergy()-state.getPotentialEnergy())/stepSize, 1e-3*abs(state.getPotentialEnergy()));
    }
}

int main() {
    try {
        testOBC(GBSAOBCForce::NoCutoff, CustomGBForce::NoCutoff);
        testOBC(GBSAOBCForce::CutoffNonPeriodic, CustomGBForce::CutoffNonPeriodic);
        testOBC(GBSAOBCForce::CutoffPeriodic, CustomGBForce::CutoffPeriodic);
        testTabulatedFunction(true);
        testTabulatedFunction(false);
        testMultipleChainRules();
        testPositionDependence();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
