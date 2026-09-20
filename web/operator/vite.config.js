import {defineConfig} from 'vite';
export default defineConfig({build:{rollupOptions:{output:{manualChunks:{three:['three']}}}},server:{allowedHosts:['.e2b.app'],proxy:{'/api':'http://127.0.0.1:8000','/ws':{target:'ws://127.0.0.1:8000',ws:true}}}});
