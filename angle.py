#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jianzhou Zhao
LastEditors: your name
@Description: File content
@Date: 2016-09-12 09:00:19
LastEditTime: 2024-03-12 16:03:31
'''

import numpy as np
from numpy.linalg import norm

np.set_printoptions(precision=8, sign="+", suppress=True, floatmode='fixed')

latt1 = np.array(([[9.5764790000,0.0000000000,0.0000000000],
                   [0.0000000000,9.5764790000,0.0000000000],
                   [0.0000000000,0.0000000000,9.5764790000]]), dtype=float)

atom1 = np.array([ 0.66565,  0.00245,  0.70867], dtype=float)
atom2 = np.array([ 0.50000, -0.13668,  0.50000], dtype=float)
atom3 = np.array([ 0.65779, -0.00000,  0.50000], dtype=float)

atom1 = np.dot(atom1, latt1)
atom2 = np.dot(atom2, latt1)
atom3 = np.dot(atom3, latt1)

oa = norm(atom1-atom3)
ob = norm(atom2-atom3)

print("oa, ob", oa, ob)

vec1 = atom1-atom3
vec2 = atom2-atom3

print(atom1, atom2, atom3)
print(vec1, vec2)

angle1 = np.arccos(np.dot(vec1, vec2)/(norm(vec1)*norm(vec2)))

vec1 = atom2-atom1
vec2 = atom3-atom1

angle2 = np.arccos(np.dot(vec1, vec2)/(norm(vec1)*norm(vec2)))

angle3 = angle1-np.pi/2.0

print(angle1*180.0/np.pi, angle2*180.0/np.pi, angle3*180.0/np.pi)

oc = oa*np.tan(angle2)

print("oc", oc)

amat = np.zeros((3, 3), dtype=float)
bmat = np.zeros(3, dtype=float)

amat[0, 0] = +(atom2[1]-atom1[1])
amat[0, 1] = -(atom2[0]-atom1[0])
amat[1, 0] = +(atom2[2]-atom1[2])
amat[1, 2] = -(atom2[0]-atom1[0])
amat[2, 0] = +(atom2[0]-atom3[0])
amat[2, 1] = +(atom2[1]-atom3[1])
amat[2, 2] = +(atom2[2]-atom3[2])

bmat[0] = (atom2[1]-atom1[1])*atom1[0]-(atom2[0]-atom1[0])*atom1[1]
bmat[1] = (atom2[2]-atom1[2])*atom1[0]-(atom2[0]-atom1[0])*atom1[2]
bmat[2] = (atom2[0]-atom3[0])*atom3[0]+(atom2[1]-atom3[1])*atom3[1] \
    + (atom2[2]-atom3[2])*atom3[2] + oc*ob*np.cos(angle3)

print(amat)
print(bmat)

x = np.linalg.solve(amat, bmat)

print()
print('x point :')
print(x)
print(np.dot(x, np.linalg.inv(latt1)))

vecz = atom1-atom3
vecx = x - atom3
vecy = np.cross(vec1, vec2)

anglep = np.arccos(np.dot(vecx, vecy)/(norm(vecx)*norm(vecy)))

print(anglep*180.0/np.pi)

print(np.dot(vecx, vecy))

print()
print("Final X:")
print(vecx)
print("Final Y:")
print(vecy)
print("Final Z:")
print(vecz)

print()
print("Final X (Normalized):")
print(vecx/norm(vecx))
print("Final Y (Normalized):")
print(vecy/norm(vecy))
print("Final Z (Normalized):")
print(vecz/norm(vecz))
