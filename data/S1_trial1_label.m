% 0:Stay, 1:Walking, 2:Swing, 3:Transition
S1_trial1 = 3*ones(39750,1);
S1_trial1(1:2722) = 0;
S1_trial1(2722:4204) = 1;
S1_trial1(5037:5650) = 2;
S1_trial1(6174:8013) = 1;
S1_trial1(8990:9745) = 2;
S1_trial1(10150:11860) = 1;
S1_trial1(12820:13800) = 2;
S1_trial1(14270:15430) = 1;
S1_trial1(15830:16940) = 0;
S1_trial1(16940:19490) = 1;
S1_trial1(19830:21190) = 0;
S1_trial1(21190:22640) = 1;
S1_trial1(23390:24220) = 2;
S1_trial1(24540:26310) = 1;
S1_trial1(27010:27990) = 2;
S1_trial1(28350:30270) = 1;
S1_trial1(31130:32370) = 2;
S1_trial1(32870:33960) = 1;
S1_trial1(34260:35560) = 0;
S1_trial1(35560:38170) = 1;
S1_trial1(38170:39610) = 0;
save S1_trial1_label.txt S1_trial1 -ascii -tabs -double


