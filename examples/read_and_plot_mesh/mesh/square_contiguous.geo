// Gmsh project created on Wed Feb  8 03:33:51 2017
Point(1) = {-1, 1, 0, 1.0};
Point(2) = {-1, 1, 0, 1.0};
Point(3) = {-1, 1, 0, 1.0};
Point(4) = {-1, 1, 0, 1.0};
Point(5) = {-1, 1, 0, 1.0};
Point(6) = {-1, 1, 0, 1.0};
Point(7) = {-1, 1, 0, 1.0};
Point(8) = {-1, 1, 0, 1.0};
Delete {
  Point{1};
}
Delete {
  Point{2};
}
Delete {
  Point{3};
}
Delete {
  Point{4};
}
Delete {
  Point{5};
}
Delete {
  Point{6};
}
Delete {
  Point{7};
}
Delete {
  Point{8};
}
Point(8) = {-1, 1, 0, 1.0};
Point(9) = {1, 1, 0, 1.0};
Point(10) = {1, -1, 0, 1.0};
Point(11) = {-1, -1, 0, 1.0};
Line(1) = {8, 11};
Line(2) = {11, 10};
Line(3) = {10, 9};
Line(4) = {9, 8};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
Transfinite Line {4, 2} = 11 Using Progression 1;
Transfinite Line {1, 3} = 11 Using Progression 1;
Transfinite Surface {6};
Recombine Surface {6};
