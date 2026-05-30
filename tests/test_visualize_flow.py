import numpy as np
import pytest
from visualize_flow import generate_flow_data

def test_generate_flow_data():
    res = 10
    u0 = 1.0
    X, Y, U, V, Vort = generate_flow_data(res, u0)
    
    # Check shapes
    assert X.shape == (res, res)
    assert Y.shape == (res, res)
    assert U.shape == (res, res)
    assert V.shape == (res, res)
    assert Vort.shape == (res, res)
    
    # Check specific values at origin (x=0, y=0)
    # Vorticity: u0 * cos(0) * cos(0) = u0
    assert np.isclose(Vort[0, 0], u0)
    # U velocity: -0.5 * u0 * cos(0) * sin(0) = 0
    assert np.isclose(U[0, 0], 0)
    # V velocity: 0.5 * u0 * sin(0) * cos(0) = 0
    assert np.isclose(V[0, 0], 0)
