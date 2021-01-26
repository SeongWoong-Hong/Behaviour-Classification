% 0:Stay, 1:Walking, 2:Swing, 3:Transition
S1_trial2 = 3*ones(39700,1);
S1_trial2(561:2845) = 0;
S1_trial2(2845:4149) = 1;
S1_trial2(5011:6096) = 2;
S1_trial2(6466:8277) = 1;
S1_trial2(9469:9972) = 2;
S1_trial2(10520:12340) = 1;
S1_trial2(13710:14080) = 2;
S1_trial2(14570:15670) = 1;
S1_trial2(16000:17370) = 0;
S1_trial2(17370:19820) = 1;
S1_trial2(20310:21480) = 0;
S1_trial2(21480:22880) = 1;
S1_trial2(23980:24570) = 2;
S1_trial2(25130:26920) = 1;
S1_trial2(27860:28540) = 2;
S1_trial2(28950:30920) = 1;
S1_trial2(31960:32570) = 2;
S1_trial2(32960:34150) = 1;
S1_trial2(34440:35660) = 0;
S1_trial2(35660:38370) = 1;
S1_trial2(38370:39510) = 0;
save S1_trial2_label.txt S1_trial2 -ascii -tabs -double