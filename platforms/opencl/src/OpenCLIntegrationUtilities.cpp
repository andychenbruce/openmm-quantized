/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2020 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "OpenCLIntegrationUtilities.h"
#include "OpenCLContext.h"

using namespace OpenMM;
using namespace std;

OpenCLIntegrationUtilities::OpenCLIntegrationUtilities(OpenCLContext& context, const System& system) : IntegrationUtilities(context, system) {
        ccmaConvergedHostBuffer.initialize<cl_int>(context, 1, "CcmaConvergedHostBuffer", CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR);
        // Different communication mechanisms give optimal performance on AMD and on NVIDIA.
        string vendor = context.getDevice().getInfo<CL_DEVICE_VENDOR>();
        ccmaUseDirectBuffer = (vendor.size() >= 28 && vendor.substr(0, 28) == "Advanced Micro Devices, Inc.");
}

OpenCLArray& OpenCLIntegrationUtilities::getPosDelta() {
    return dynamic_cast<OpenCLContext&>(context).unwrap(posDelta);
}

OpenCLArray& OpenCLIntegrationUtilities::getRandom() {
    return dynamic_cast<OpenCLContext&>(context).unwrap(random);
}

OpenCLArray& OpenCLIntegrationUtilities::getStepSize() {
    return dynamic_cast<OpenCLContext&>(context).unwrap(stepSize);
}

void OpenCLIntegrationUtilities::applyConstraintsImpl(bool constrainVelocities, double tol) {
    ComputeKernel settleKernel, shakeKernel, ccmaForceKernel;
    if (constrainVelocities) {
        settleKernel = settleVelKernel;
        shakeKernel = shakeVelKernel;
        ccmaForceKernel = ccmaVelForceKernel;
    }
    else {
        settleKernel = settlePosKernel;
        shakeKernel = shakePosKernel;
        ccmaForceKernel = ccmaPosForceKernel;
    }
    if (settleAtoms.isInitialized()) {
      switch (context.getPrecision()){
      case PrecisionLevel::Double :
      case PrecisionLevel::Mixed :
	{
	  settleKernel->setArg(1, tol);
	  break;
	}
      case PrecisionLevel::Single :
	{
	  settleKernel->setArg(1, (float) tol);
	  break;
	}
      case PrecisionLevel::F16 :
	{
	  settleKernel->setArg(1, (half) tol);
	  break;
	}
      }
        settleKernel->execute(settleAtoms.getSize());
    }
    if (shakeAtoms.isInitialized()) {
      switch (context.getPrecision()){
      case PrecisionLevel::Double :
      case PrecisionLevel::Mixed :
	{
	  shakeKernel->setArg(1, tol);
	  break;
	}
      case PrecisionLevel::Single :
	{
	  shakeKernel->setArg(1, (float) tol);
	  break;
	}
      case PrecisionLevel::F16 :
	{
	  shakeKernel->setArg(1, (half) tol);
	  break;
	}
      }
        shakeKernel->execute(shakeAtoms.getSize());
    }
    if (ccmaConstraintAtoms.isInitialized()) {
        if (ccmaConstraintAtoms.getSize() <= 1024) {
            // Use the version of CCMA that runs in a single kernel with one workgroup.
            ccmaFullKernel->setArg(0, (int) constrainVelocities);
            switch (context.getPrecision()){
	    case PrecisionLevel::Double :
	    case PrecisionLevel::Mixed :
	      {
		ccmaFullKernel->setArg(14, tol);
		break;
	      }
	    case PrecisionLevel::Single :
	      {
		ccmaFullKernel->setArg(14, (float) tol);
		break;
	      }
	    case PrecisionLevel::F16 :
	      {
		ccmaFullKernel->setArg(14, (half) tol);
		break;
	      }
	    }
            ccmaFullKernel->execute(128, 128);
        }
        else {
            // Use the version of CCMA that uses multiple kernels.
            ccmaForceKernel->setArg(6, ccmaConvergedHostBuffer);
            switch (context.getPrecision()){
	    case PrecisionLevel::Double :
	    case PrecisionLevel::Mixed :
	      {
		ccmaForceKernel->setArg(7, tol);
		break;
	      }
	    case PrecisionLevel::Single :
	      {
		ccmaForceKernel->setArg(7, (float) tol);
		break;
	      }
	    case PrecisionLevel::F16 :
	      {
		ccmaForceKernel->setArg(7, (half) tol);
		break;
	      }
	    }
            ccmaDirectionsKernel->execute(ccmaConstraintAtoms.getSize());
            const int checkInterval = 4;
            OpenCLContext& cl = dynamic_cast<OpenCLContext&>(context);
            cl::CommandQueue queue = cl.getQueue();
            int* converged = (int*) context.getPinnedBuffer();
            int* ccmaConvergedHostMemory = (int*) queue.enqueueMapBuffer(ccmaConvergedHostBuffer.getDeviceBuffer(), CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_int));
            ccmaConvergedHostMemory[0] = 0;
            queue.enqueueUnmapMemObject(ccmaConvergedHostBuffer.getDeviceBuffer(), ccmaConvergedHostMemory);
            ccmaUpdateKernel->setArg(4, constrainVelocities ? context.getVelm() : posDelta);
            for (int i = 0; i < 150; i++) {
                ccmaForceKernel->setArg(8, i);
                ccmaForceKernel->execute(ccmaConstraintAtoms.getSize());
                cl::Event event;
                if ((i+1)%checkInterval == 0 && !ccmaUseDirectBuffer)
                    queue.enqueueReadBuffer(cl.unwrap(ccmaConverged).getDeviceBuffer(), CL_FALSE, 0, 2*sizeof(int), converged, NULL, &event);
                ccmaMultiplyKernel->setArg(5, i);
                ccmaMultiplyKernel->execute(ccmaConstraintAtoms.getSize());
                ccmaUpdateKernel->setArg(9, i);
                ccmaUpdateKernel->execute(context.getNumAtoms());
                if ((i+1)%checkInterval == 0) {
                    if (ccmaUseDirectBuffer) {
                        ccmaConvergedHostMemory = (int*) queue.enqueueMapBuffer(ccmaConvergedHostBuffer.getDeviceBuffer(), CL_FALSE, CL_MAP_READ, 0, sizeof(cl_int), NULL, &event);
                        queue.flush();
                        while (event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() != CL_COMPLETE)
                            ;
                        converged[i%2] = ccmaConvergedHostMemory[0];
                        queue.enqueueUnmapMemObject(ccmaConvergedHostBuffer.getDeviceBuffer(), ccmaConvergedHostMemory);
                    }
                    else
                        event.wait();
                    if (converged[i%2])
                        break;
                }
            }
        }
    }
}

void OpenCLIntegrationUtilities::distributeForcesFromVirtualSites() {
    for (int i = numVsiteStages-1; i >= 0; i--) {
        vsiteForceKernel->setArg(2, context.getLongForceBuffer());
        vsiteForceKernel->setArg(15, i);
        vsiteForceKernel->execute(numVsites);
        vsiteSaveForcesKernel->setArg(0, context.getLongForceBuffer());
        vsiteSaveForcesKernel->setArg(1, context.getForceBuffers());
        vsiteSaveForcesKernel->execute(context.getNumAtoms());
   }
}
