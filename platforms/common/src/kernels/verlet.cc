/**
 * Perform the first step of Verlet integration.
 */
KERNEL void integrateVerletPart1(int numAtoms, int paddedNumAtoms, GLOBAL const mixed2* RESTRICT dt, GLOBAL const real4* RESTRICT posq,
        GLOBAL mixed4* RESTRICT velm, GLOBAL const mm_long* RESTRICT force, GLOBAL mixed4* RESTRICT posDelta
#ifdef USE_MIXED_PRECISION
        , GLOBAL const real4* RESTRICT posqCorrection
#endif
				 ) {
  // for(int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE){
  //   printf("BEFORE PART1 %d is = %f, %f, %f, %f\n", i, (double)posq[i].x, (double)posq[i].y, (double)posq[i].z, (double)posq[i].w);
  // }

  
    const mixed2 stepSize = dt[0];
    const mixed dtPos = stepSize.y;
    const mixed dtVel = (mixed)0.5*(stepSize.x+stepSize.y);
    const mixed scale = dtVel/(mixed) (long long) 0x100000000;
    for (int index = GLOBAL_ID; index < numAtoms; index += GLOBAL_SIZE) {
        mixed4 velocity = velm[index];
	    printf("velm before at %f, %f, %f, %f\n",
		   (double) velocity.x,
		   (double) velocity.y,
		   (double) velocity.z,
		   (double) velocity.w);
        if (velocity.w != (mixed)0.0) {
#ifdef USE_MIXED_PRECISION
            real4 pos1 = posq[index];
            real4 pos2 = posqCorrection[index];
            mixed4 pos = make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
#else
            real4 pos = posq[index];
#endif
	    printf("force mul %d, %d, %d is %lld, %lld, %lld\n", index, index+paddedNumAtoms, index+paddedNumAtoms*2, force[index], force[index+paddedNumAtoms], force[index+paddedNumAtoms*2]);
            velocity.x += scale*(mixed)force[index]*velocity.w;
            velocity.y += scale*(mixed)force[index+paddedNumAtoms]*velocity.w;
            velocity.z += scale*(mixed)force[index+paddedNumAtoms*2]*velocity.w;
	    printf("velm after at %f, %f, %f, %f\n",
		   (double) velocity.x,
		   (double) velocity.y,
		   (double) velocity.z,
		   (double) velocity.w);
            pos.x = velocity.x*dtPos;
            pos.y = velocity.y*dtPos;
            pos.z = velocity.z*dtPos;
            posDelta[index] = pos;
            velm[index] = velocity;
        }
    }

  // for(int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE){
  //   printf("AFTER PART1 %d is = %f, %f, %f, %f\n", i, (double)posq[i].x, (double)posq[i].y, (double)posq[i].z, (double)posq[i].w);
  // }

}

/**
 * Perform the second step of Verlet integration.
 */

KERNEL void integrateVerletPart2(int numAtoms, GLOBAL mixed2* RESTRICT dt, GLOBAL real4* RESTRICT posq,
        GLOBAL mixed4* RESTRICT velm, GLOBAL const mixed4* RESTRICT posDelta
#ifdef USE_MIXED_PRECISION
        , GLOBAL real4* RESTRICT posqCorrection
#endif
				 ) {
  // for(int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE){
  //   printf("BEFORE PART2 %d is = %f, %f, %f, %f\n", i, (double)posq[i].x, (double)posq[i].y, (double)posq[i].z, (double)posq[i].w);
  // }
  
    mixed2 stepSize = dt[0];
    //printf("step size = %f, %f\n", (double) stepSize.x, (double) stepSize.y);
#ifdef SUPPORTS_DOUBLE_PRECISION
    double oneOverDt = 1.0/(double)stepSize.y;
#else
    float oneOverDt = 1.0f/(float)stepSize.y;
    float correction = (1.0f-oneOverDt*stepSize.y)/stepSize.y;
#endif
    if (GLOBAL_ID == 0)
        dt[0].x = stepSize.y;
    SYNC_THREADS;
    int index = GLOBAL_ID;
    for (; index < numAtoms; index += GLOBAL_SIZE) {
        mixed4 velocity = velm[index];
      	printf("PART2 velm before at %f, %f, %f, %f\n",
	       (double) velocity.x,
	       (double) velocity.y,
	       (double) velocity.z,
	       (double) velocity.w);

        if (velocity.w != (mixed)0.0) {
#ifdef USE_MIXED_PRECISION
            real4 pos1 = posq[index];
            real4 pos2 = posqCorrection[index];
            mixed4 pos = make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
#else
            real4 pos = posq[index];
#endif
            mixed4 delta = posDelta[index];
            pos.x += delta.x;
            pos.y += delta.y;
            pos.z += delta.z;
	    printf("one over dt = %f\n", (double) oneOverDt);
	    printf("delta %d = %f, %f, %f, %f\n", index,
		   (double)delta.x,
		   (double)delta.y,
		   (double)delta.z,
		   (double)delta.w);
#ifdef SUPPORTS_DOUBLE_PRECISION
            velocity = make_mixed4((mixed) ((float)delta.x*oneOverDt), (mixed) ((float)delta.y*oneOverDt), (mixed) ((float)delta.z*oneOverDt), velocity.w);
#else
            velocity = make_mixed4((mixed) (delta.x*oneOverDt+delta.x*correction), (mixed) (delta.y*oneOverDt+delta.y*correction), (mixed) (delta.z*oneOverDt+delta.z*correction), velocity.w);
#endif
#ifdef USE_MIXED_PRECISION
            posq[index] = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) pos.w);
            posqCorrection[index] = make_real4(pos.x-(real) pos.x, pos.y-(real) pos.y, pos.z-(real) pos.z, 0);
#else
            posq[index] = pos;
#endif
            velm[index] = velocity;
      	    printf("PART2 velm after at %f, %f, %f, %f\n",
	       (double) velocity.x,
	       (double) velocity.y,
	       (double) velocity.z,
	       (double) velocity.w);
        }

    }

    // for(int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE){
    //   printf("AFTER PART2 %d is = %f, %f, %f, %f\n", i, (double)posq[i].x, (double)posq[i].y, (double)posq[i].z, (double)posq[i].w);
    // }
}

/**
 * Select the step size to use for the next step.
 */

KERNEL void selectVerletStepSize(int numAtoms, int paddedNumAtoms, mixed maxStepSize, mixed errorTol, GLOBAL mixed2* RESTRICT dt, GLOBAL const mixed4* RESTRICT velm, GLOBAL const mm_long* RESTRICT force) {
    // Calculate the error.

    LOCAL mixed error[256];
    mixed err = 0;
    const mixed scale = RECIP((float) 0x100000000);
    for (int index = LOCAL_ID; index < numAtoms; index += LOCAL_SIZE) {
        mixed3 f = make_mixed3(scale*(mixed)force[index], scale*(mixed)force[index+paddedNumAtoms], scale*(mixed)force[index+paddedNumAtoms*2]);
        mixed invMass = velm[index].w;
        err += (f.x*f.x + f.y*f.y + f.z*f.z)*invMass*invMass;
    }
    error[LOCAL_ID] = err;
    SYNC_THREADS;

    // Sum the errors from all threads.

    for (unsigned int offset = 1; offset < LOCAL_SIZE; offset *= 2) {
        if (LOCAL_ID+offset < LOCAL_SIZE && (LOCAL_ID&(2*offset-1)) == 0)
            error[LOCAL_ID] += error[LOCAL_ID+offset];
        SYNC_THREADS;
    }
    if (LOCAL_ID == 0) {
        mixed totalError = SQRT(error[0]/(mixed)(numAtoms*3));
        mixed newStepSize = SQRT(errorTol/totalError);
        mixed oldStepSize = dt[0].y;
        if (oldStepSize > (mixed)0.0)
            newStepSize = (mixed)min((float)newStepSize, (float)(oldStepSize*(mixed)2.0)); // For safety, limit how quickly dt can increase.
        if (newStepSize > oldStepSize && newStepSize < (mixed)1.1*oldStepSize)
            newStepSize = oldStepSize; // Keeping dt constant between steps improves the behavior of the integrator.
        if (newStepSize > maxStepSize)
            newStepSize = maxStepSize;
        dt[0].y = newStepSize;
    }
}
