// Node server to handle socket io connections
const io = require('socket.io')(8000)
const users = {};

io.on('connection',socket =>{ //listen to socket connection, i.e. any user that joins, means a socket connections is made
    socket.on('new-user-joined', name =>{ //handles a particular user(accepts a new-user-joined event)
        users[socket.id] = name;
        socket.broadcast.emit('user-joined', name) //emit to all user except current user
    });

    socket.on('send', message =>{ //for sending message (all receive, user-joined etc are events)
        socket.broadcast.emit('receive', {message:message, name:users[socket.id]})
    });

    socket.on('disconnect', message =>{ //for sending message (all receive, user-joined etc are events)
        socket.broadcast.emit('left', users[socket.id]);
        delete users[socket.id];
    });
})

