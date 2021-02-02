% 0:Stay, 1:Walking, 2:Swing, 3:Transition
S3_trial2 = 3*ones(31350,1);
S3_trial2(656:2724) = 0;
S3_trial2(2724:4104)=1;
S3_trial2(4289:4660)=2;
S3_trial2(5044:6832)=1;
S3_trial2(7071:7405)=2;
S3_trial2(7746:9423)=1;
S3_trial2(9755:10090)=2;
S3_trial2(10390:11590)=1;
S3_trial2(11660:12860)=0;
S3_trial2(12910:15440)=1;
S3_trial2(15520:16950)=0;
S3_trial2(17020:18300)=1;
S3_trial2(18600:18910)=2;
S3_trial2(19260:20940)=1;
S3_trial2(21220:21550)=2;
S3_trial2(21850:23690)=1;
S3_trial2(23990:24340)=2;
S3_trial2(24620:25900)=1;
S3_trial2(25980:27200)=0;
S3_trial2(27270:29860)=1;
S3_trial2(29960:31260)=0;
save S3_trial2_label.txt S3_trial2 -ascii -tabs -double